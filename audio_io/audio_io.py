import itertools
import json
import os
import subprocess as sp
import sys
from enum import Enum, auto
from numbers import Number
from os import path
from subprocess import DEVNULL, PIPE
from typing import NamedTuple, Iterator, Iterable, List, Optional, Sequence

import numpy as np

from audio_io.cue.cue_parser import CueCmd, parse_cue_str, read_cue_from_file
from util.natural_sort import natural_sort_key

ex_ffprobe = 'ffprobe'
ex_ffmpeg = 'ffmpeg'

known_audio_extensions = {
    'aac',
    'ac3',
    'aif',
    'aiff',
    'ape',
    'dts',
    'flac',
    'm4a',
    'mka',
    'mp2',
    'mp3',
    'mpc',
    'ofr',
    'ogg',
    'opus',
    'tak',
    'tta',
    'wav',
    'wv',
}


class FileKind(Enum):
    CUE = auto()
    FOLDER = auto()
    AUDIO = auto()


class TagKey(str, Enum):
    TITLE = 'TITLE'
    ALBUM = 'ALBUM'
    ARTIST = 'ARTIST'
    PERFORMER = 'PERFORMER'
    CUESHEET = 'CUESHEET'


_tag_alternatives = {
    TagKey.PERFORMER: (TagKey.ARTIST,),
    TagKey.ARTIST: (TagKey.PERFORMER,),
}


def get_tag_with_alternatives(tags: dict, tag_key: TagKey) -> Optional[str]:
    exact_match = tags.get(tag_key)
    if exact_match:
        return exact_match
    for alt_key in _tag_alternatives.get(tag_key, ()):
        v = tags.get(alt_key)
        if v:
            return v
    return None


def get_file_kind(in_path: str) -> FileKind:
    if os.path.isdir(in_path):
        return FileKind.FOLDER
    _, ext = path.splitext(in_path)
    if '.cue' == ext.lower():
        return FileKind.CUE
    return FileKind.AUDIO


class TrackInfo(NamedTuple):
    global_index: int
    offset_seconds: Number
    tags: dict


class AudioFileMetadata(NamedTuple):
    file_path: str
    channel_count: int
    sample_rate: int
    cuesheet: str or None
    tags: dict


class AudioSourceInfo(NamedTuple):
    file_path: str
    channel_count: int
    sample_rate: int
    tags: dict
    tracks: List[TrackInfo]


class AudioData(NamedTuple):
    source_info: AudioSourceInfo
    samples_per_block: int
    blocks_generator: Iterator[Iterator[np.ndarray]]


def _translate_from_cue(cue_items,
                        directory_path=None,
                        parent_audio_file: AudioFileMetadata = None) -> Iterable[AudioSourceInfo]:
    global_track_counter = itertools.count(1)
    index_number = None
    index_offset_seconds = None
    last_file_path = None
    channel_count = None
    sample_rate = None

    track_start = False  # if parser is between TRACK and INDEX commands
    global_tags = dict()
    track_tags = dict()

    def add_tag(key: TagKey, value: str, is_global: bool):
        if is_global:
            global_tags[key] = value
            if key not in track_tags:
                track_tags[key] = value
        else:
            track_tags[key] = value

    tracks = []

    join = os.path.join

    for cmd, *args in cue_items:
        if cmd == CueCmd.TRACK or cmd == CueCmd.FILE or cmd == CueCmd.EOF:
            if index_number is not None:
                assert index_offset_seconds is not None
                # noinspection PyTypeChecker
                tracks.append(TrackInfo(
                    global_index=next(global_track_counter),
                    offset_seconds=index_offset_seconds,
                    tags=track_tags
                ))
                track_tags = dict(global_tags)
                index_number = None
            if cmd == CueCmd.TRACK:
                track_start = True
                continue

        if cmd == CueCmd.FILE or cmd == CueCmd.EOF:
            if last_file_path:
                yield AudioSourceInfo(
                    file_path=last_file_path,
                    channel_count=channel_count,
                    sample_rate=sample_rate,
                    tracks=tracks,
                    tags=global_tags)
                tracks = []
            if cmd == CueCmd.EOF:
                return

            if directory_path:
                last_file_path = join(directory_path, args[0])
                p = read_audio_file_metadata(last_file_path)
                channel_count, sample_rate = p.channel_count, p.sample_rate
                global_tags.update(p.tags)
            elif parent_audio_file:
                last_file_path = parent_audio_file.file_path
                channel_count = parent_audio_file.channel_count
                sample_rate = parent_audio_file.sample_rate
                global_tags.update(parent_audio_file.tags)
            else:
                raise ValueError
        elif cmd == CueCmd.TITLE:
            add_tag(TagKey.TITLE, args[0], is_global=not track_start)
            if not track_start:
                add_tag(TagKey.ALBUM, args[0], is_global=True)
        elif cmd == CueCmd.PERFORMER:
            add_tag(TagKey.PERFORMER, args[0], is_global=not track_start)
        elif cmd == CueCmd.INDEX:
            track_start = False
            number, offset = args

            if len(tracks):
                num_condition = lambda: index_number < number
            else:
                num_condition = lambda: index_number > number

            if (index_number is None) or (number <= 1 and num_condition()):
                index_number, index_offset_seconds = number, offset
        elif cmd == CueCmd.REM:
            add_tag(args[0], args[1], is_global=not track_start)
        else:
            raise NotImplementedError


def _single_track_audio_source(p: AudioFileMetadata, track_index):
    track_info = TrackInfo(global_index=track_index, offset_seconds=0, tags=p.tags)
    return AudioSourceInfo(
        file_path=p.file_path,
        channel_count=p.channel_count,
        sample_rate=p.sample_rate,
        tags=p.tags,
        tracks=[track_info])


def _audio_source_from_file(in_path, track_index=1) -> AudioSourceInfo:
    p = read_audio_file_metadata(in_path)
    if not p.cuesheet:
        return _single_track_audio_source(p, track_index)
    cue_entries = parse_cue_str(p.cuesheet)
    return next(iter(_translate_from_cue(cue_entries, parent_audio_file=p)))


def _audio_sources_from_folder(in_path) -> Iterable[AudioSourceInfo]:
    track_counter = itertools.count(1)
    for dirpath, dirnames, filenames in os.walk(in_path, topdown=True):
        filenames = sorted(filenames, key=natural_sort_key)
        for f in filenames:
            _, ext = path.splitext(f)
            ext = ext[1:].lower()
            if ext in known_audio_extensions:
                yield _audio_source_from_file(path.join(in_path, f), track_index=next(track_counter))
        break


def read_audio_info(in_path: str) -> Iterable[AudioSourceInfo]:
    """
    if input file is a cue, it can reference multiple audio files with different sample rates.
    therefore the result is a sequence.
    """
    kind = get_file_kind(in_path)
    if kind == FileKind.FOLDER:
        yield from _audio_sources_from_folder(in_path)
    elif kind == FileKind.CUE:
        cue_str = read_cue_from_file(in_path)
        yield from _translate_from_cue(parse_cue_str(cue_str), directory_path=os.path.dirname(in_path))
    elif kind == FileKind.AUDIO:
        yield _audio_source_from_file(in_path)
    else:
        raise NotImplementedError


def read_audio_data(audio_source: AudioSourceInfo,
                    samples_per_block: int,
                    ffmpeg_args: Sequence[str],
                    bytes_per_sample_mono: int,
                    numpy_sample_type: str,
                    sample_rate: Optional[int] = None) -> AudioData:
    audio_blocks = _read_audio_blocks(audio_source,
                                      samples_per_block,
                                      ffmpeg_args,
                                      bytes_per_sample_mono,
                                      numpy_sample_type,
                                      sample_rate)
    return AudioData(audio_source, samples_per_block, audio_blocks)


def _test_ffmpeg():
    try:
        for n in (ex_ffmpeg, ex_ffprobe):
            sp.check_call((n, '-version'), stderr=DEVNULL, stdout=DEVNULL)
    except sp.CalledProcessError:
        sys.exit('ffmpeg not installed, broken or not on PATH')


def _parse_audio_metadata(in_path: str, data_from_ffprobe: dict) -> AudioFileMetadata:
    def get(*keys, default_value=None):
        d = data_from_ffprobe
        for k in keys:
            try:
                d = d[k]
            except (KeyError, IndexError):
                return default_value
        return d

    tags = {key.upper(): val for key, val in get('format', 'tags', default_value={}).items()}
    return AudioFileMetadata(
        file_path=in_path,
        channel_count=int(get('streams', 0, 'channels')),
        sample_rate=int(get('streams', 0, 'sample_rate')),
        tags=tags,
        cuesheet=tags.get(TagKey.CUESHEET))


def read_audio_file_metadata(in_path) -> AudioFileMetadata:
    p = sp.Popen(
        (ex_ffprobe,
         '-v', 'error',
         '-print_format', 'json',
         '-select_streams', 'a:0',
         '-show_entries', 'stream=channels,sample_rate',
         '-show_entries', 'format_tags',
         in_path),
        stdout=PIPE, stderr=PIPE)
    out, err = p.communicate()
    returncode = p.returncode
    if returncode != 0:
        raise Exception('ffprobe returned {}'.format(returncode))
    audio_metadata = _parse_audio_metadata(in_path, json.loads(out, encoding='utf-8'))
    assert audio_metadata.channel_count >= 1
    return audio_metadata


def _read_audio_blocks(audio_source: AudioSourceInfo,
                       samples_per_block: int,
                       ffmpeg_args: Sequence[str],
                       bytes_per_sample_mono: int,
                       numpy_sample_type: str,
                       sample_rate: Optional[int] = None) -> Iterator[Iterator[np.ndarray]]:
    bytes_per_sample = bytes_per_sample_mono * audio_source.channel_count
    max_bytes_per_block = bytes_per_sample * samples_per_block
    sample_rate = sample_rate or audio_source.sample_rate

    # noinspection PyTypeChecker
    def seconds_to_samples(seconds: Number):
        return int(sample_rate * seconds)

    # noinspection PyTypeChecker
    p = sp.Popen((ex_ffmpeg,) + ffmpeg_args, stderr=None, stdout=PIPE)

    sample_type = np.dtype(numpy_sample_type)
    frombuffer = np.frombuffer
    reshape = np.reshape
    with p.stdout as f:
        skip_samples = seconds_to_samples(audio_source.tracks[0].offset_seconds)
        if skip_samples > 0:
            f.read(bytes_per_sample * skip_samples)

        def make_array(buffer, size):
            a = frombuffer(buffer, dtype=sample_type, count=size // bytes_per_sample_mono)
            a = reshape(a, (audio_source.channel_count, -1), order='F')
            return a

        def read_n_bytes(n):
            while (n is None) or (n >= max_bytes_per_block):
                b = f.read(max_bytes_per_block)
                read_size = len(b)
                if read_size > 0:
                    yield make_array(b, read_size)
                    if n:
                        n -= read_size
                else:
                    return
            if n:
                b = f.read(n)
                read_size = len(b)
                assert read_size == n
                yield make_array(b, read_size)

        track_count = len(audio_source.tracks)
        for ti in range(track_count):
            if track_count == ti + 1:
                bytes_to_read = None
            else:
                samples_to_read = seconds_to_samples(audio_source.tracks[ti + 1].offset_seconds) \
                                  - seconds_to_samples(audio_source.tracks[ti].offset_seconds)
                bytes_to_read = samples_to_read * bytes_per_sample
            yield read_n_bytes(bytes_to_read)
