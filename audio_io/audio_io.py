import itertools
import os
import re
import subprocess as sp
import sys
from collections import defaultdict
from fractions import Fraction
from io import BufferedIOBase
from numbers import Number
from os import path
from subprocess import DEVNULL, PIPE
from typing import NamedTuple, Iterator, Sequence, Iterable, List

import chardet
import numpy as np

from util.constants import MEASURE_SAMPLE_RATE
from util.natural_sort import natural_sort_key

ex_ffprobe = 'ffprobe'
ex_ffmpeg = 'ffmpeg'

from enum import Enum, auto

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


class CueCmd(Enum):
    PERFORMER = auto()
    TITLE = auto()
    FILE = auto()
    TRACK = auto()
    INDEX = auto()
    EOF = auto()


def get_file_kind(in_path: str) -> FileKind:
    if os.path.isdir(in_path):
        return FileKind.FOLDER
    _, ext = path.splitext(in_path)
    if '.cue' == ext.lower():
        return FileKind.CUE
    return FileKind.AUDIO


class TrackInfo(NamedTuple):
    global_index: int
    name: str
    offset_samples: int


class AudioFileParams(NamedTuple):
    channel_count: int
    sample_rate: int
    title: str
    artist: str
    album: str


class AudioSourceInfo(NamedTuple):
    path: str
    name: str
    performers: Sequence[str]
    album: str
    channel_count: int
    sample_rate: int
    tracks: List[TrackInfo]


class AudioSource(NamedTuple):
    source_info: AudioSourceInfo
    samples_per_block: int
    blocks_generator: Iterator[Iterator[np.ndarray]]


_whitespace_pattern = re.compile('\s+')


def _unquote(s: str):
    return s[1 + s.index('"'):s.rindex('"')]


def parse_cd_time(offset: str) -> Number:
    """parse time in CDDA (75fps) format to seconds, exactly
    MM:SS:FF"""
    m, s, f = map(int, offset.split(':'))
    return m * 60 + s + Fraction(f, 75)


def _parse_cue_cmd(line: str):
    line = line.strip()
    cmd, args = _whitespace_pattern.split(line, 1)
    if cmd == 'PERFORMER':
        return CueCmd.PERFORMER, _unquote(args)
    if cmd == 'TITLE':
        return CueCmd.TITLE, _unquote(args)
    if cmd == 'FILE':
        return CueCmd.FILE, _unquote(args)
    if cmd == 'TRACK':
        number, _ = _whitespace_pattern.split(args, 1)
        number = int(number)
        return CueCmd.TRACK, number
    if cmd == 'INDEX':
        number, offset = _whitespace_pattern.split(args, 1)
        number = int(number)
        offset = parse_cd_time(offset)
        return CueCmd.INDEX, number, offset

    return None


def parse_cue(in_path):
    with open(in_path, 'rb') as f:
        assert isinstance(f, BufferedIOBase)
        content = f.read()
    encoding = chardet.detect(content)['encoding']
    content_text = content.decode(encoding)
    for line in content_text.splitlines():
        cmd = _parse_cue_cmd(line)
        if cmd:
            yield cmd
    yield CueCmd.EOF, None


def _translate_from_cue(directory_path, cue_items) -> Iterable[AudioSourceInfo]:
    global_track_counter = itertools.count(1)
    index_number = None
    index_offset = None
    last_file_path = None
    channel_count = None
    sample_rate = None

    track_start = False  # if parser is between TRACK and INDEX commands
    last_title_file = None
    last_title_track = None
    file_performer = None

    tracks = []

    join = os.path.join

    for cmd, *args in cue_items:
        if cmd == CueCmd.TRACK or cmd == CueCmd.FILE or cmd == CueCmd.EOF:
            if index_number is not None:
                assert last_title_track is not None
                assert index_offset is not None
                # noinspection PyTypeChecker
                tracks.append(TrackInfo(
                    global_index=next(global_track_counter),
                    name=last_title_track,
                    offset_samples=index_offset))
                index_number = None
            if cmd == CueCmd.TRACK:
                track_start = True
                continue

        if cmd == CueCmd.FILE or cmd == CueCmd.EOF:
            if last_file_path:
                yield AudioSourceInfo(
                    path=last_file_path,
                    name=last_title_file,
                    album=last_title_file,
                    performers=[file_performer] if file_performer else [],
                    channel_count=channel_count,
                    sample_rate=sample_rate,
                    tracks=tracks)
                tracks = []
            if cmd == CueCmd.EOF:
                return

            last_file_path = join(directory_path, args[0])
            p = _get_audio_properties(last_file_path)
            channel_count, sample_rate = p.channel_count, p.sample_rate
        elif cmd == CueCmd.TITLE:
            if track_start:
                last_title_track = args[0]
            else:
                last_title_file = args[0]
        elif cmd == CueCmd.PERFORMER:
            if not track_start:
                file_performer = args[0]
        elif cmd == CueCmd.INDEX:
            track_start = False
            number, offset = args

            if len(tracks):
                num_condition = lambda: index_number < number
            else:
                num_condition = lambda: index_number > number

            if (index_number is None) or (number <= 1 and num_condition()):
                index_number, index_offset = number, int(MEASURE_SAMPLE_RATE * offset)
        else:
            raise NotImplementedError


def _audio_source_from_file(in_path, track_index=1) -> AudioSourceInfo:
    p = _get_audio_properties(in_path)
    track_info = TrackInfo(global_index=track_index, name=p.title, offset_samples=0)
    return AudioSourceInfo(
        path=in_path,
        name=p.title,
        performers=(p.artist,),
        album=p.album,
        channel_count=p.channel_count,
        sample_rate=p.sample_rate,
        tracks=[track_info])


def _audio_sources_from_folder(in_path) -> Iterable[AudioSourceInfo]:
    track_counter = itertools.count(1)
    for dirpath, dirnames, filenames in os.walk(in_path, topdown=True):
        filenames = sorted(filenames, key=natural_sort_key)
        for f in filenames:
            _, ext = path.splitext(f)
            ext = ext[1:].lower()
            if ext in known_audio_extensions:
                yield _audio_source_from_file(path.join(in_path, f), track_index=next(track_counter))
            pass
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
        directory_path = os.path.dirname(in_path)
        yield from _translate_from_cue(directory_path, parse_cue(in_path))
    elif kind == FileKind.AUDIO:
        yield _audio_source_from_file(in_path)
    else:
        raise NotImplementedError


def _get_audio_properties(in_path) -> AudioFileParams:
    r = _get_params(in_path)
    if r.channel_count < 1 or r.sample_rate < 8000:
        sys.exit(f'invalid format: channels={r.channel_count}, sample_rate={r.sample_rate}')

    return r


def read_audio_data(what: AudioSourceInfo, samples_per_block: int) -> AudioSource:
    audio_blocks = _read_audio_blocks(what.path, what.channel_count, samples_per_block, what.tracks)
    return AudioSource(what, samples_per_block, audio_blocks)


def _test_ffmpeg():
    try:
        for n in (ex_ffmpeg, ex_ffprobe):
            sp.check_call((n, '-version'), stderr=DEVNULL, stdout=DEVNULL)
    except sp.CalledProcessError:
        sys.exit('ffmpeg not installed, broken or not on PATH')


def _parse_audio_params(s) -> AudioFileParams:
    d = defaultdict(lambda: '(unknown)')
    for m in re.finditer(r'([a-zA-Z_:]+)=(.+)', s):
        v = m.groups()
        d[v[0].lower()] = v[1]

    return AudioFileParams(
        int(d['channels']),
        int(d['sample_rate']),
        d['tag:title'],
        d['tag:artist'],
        d['tag:album'])


def _get_params(in_path) -> AudioFileParams:
    p = sp.Popen(
        (ex_ffprobe,
         '-v', 'error',
         '-select_streams', '0:a:0',
         '-show_entries', 'stream=channels,sample_rate',
         '-show_entries', 'format_tags=title,artist,album',
         in_path),
        stdout=PIPE)
    out, err = p.communicate()
    returncode = p.returncode
    if returncode != 0:
        raise Exception('ffprobe returned {}'.format(returncode))
    out = out.decode('utf-8')
    return _parse_audio_params(out)


def _read_audio_blocks(in_path, channel_count, samples_per_block, tracks: List[TrackInfo]) -> \
        Iterator[Iterator[np.ndarray]]:
    bytes_per_sample = 4 * channel_count
    max_bytes_per_block = bytes_per_sample * samples_per_block

    p = sp.Popen(
        (ex_ffmpeg, '-loglevel', 'fatal',
         '-i', in_path,
         '-map', '0:a:0',
         '-c:a', 'pcm_f32le',
         '-ar', str(MEASURE_SAMPLE_RATE),
         # ^ because apparently official meter resamples to 44k before measuring;
         # using default low quality resampling because it doesn't affect measurements and is faster
         '-f', 'f32le',
         '-'),
        stderr=None,
        stdout=PIPE)

    sample_type = np.dtype('<f4')
    frombuffer = np.frombuffer
    reshape = np.reshape
    with p.stdout as f:
        skip_samples = tracks[0].offset_samples
        if skip_samples > 0:
            f.read(bytes_per_sample * skip_samples)

        def make_array(buffer, size):
            a = frombuffer(buffer, dtype=sample_type, count=size // 4)
            a = reshape(a, (channel_count, -1), order='F')
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

        track_count = len(tracks)
        for ti in range(track_count):
            if track_count == ti + 1:
                bytes_to_read = None
            else:
                samples_to_read = tracks[ti + 1].offset_samples - tracks[ti].offset_samples
                bytes_to_read = samples_to_read * bytes_per_sample
            yield read_n_bytes(bytes_to_read)
