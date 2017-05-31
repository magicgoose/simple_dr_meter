import re
import subprocess as sp
from collections import OrderedDict
from fractions import Fraction
from io import SEEK_CUR
from numbers import Number

import chardet

import sys
from subprocess import DEVNULL, PIPE
from typing import NamedTuple, Iterator, Sequence, Iterable, List

import numpy as np
from os import path

import os

ex_ffprobe = 'ffprobe'
ex_ffmpeg = 'ffmpeg'

from enum import Enum, auto


class FileKind(Enum):
    CUE = auto()
    AUDIO = auto()


class CueCmd(Enum):
    PERFORMER = auto()
    TITLE = auto()
    FILE = auto()
    TRACK = auto()
    INDEX = auto()
    EOF = auto()


def get_file_kind(in_path: str) -> FileKind:
    _, ext = path.splitext(in_path)
    if '.cue' == ext.lower():
        return FileKind.CUE
    return FileKind.AUDIO


class TrackInfo(NamedTuple):
    name: str
    offset_samples: int


class AudioSourceInfo(NamedTuple):
    path: str
    name: str
    performers: Sequence[str]
    channel_count: int
    sample_rate: int
    tracks: Sequence[TrackInfo]


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
    """returns all entries of CUE and True when done"""
    # detect file encoding
    with open(in_path, 'rb') as f:
        raw = f.read(32)  # at most 32 bytes are returned
        encoding = chardet.detect(raw)['encoding']
    with open(in_path, 'r', encoding=encoding) as f:
        for line in f:
            cmd = _parse_cue_cmd(line)
            if cmd:
                yield cmd
        yield CueCmd.EOF, None


def _translate_from_cue(directory_path, cue_items) -> Iterable[AudioSourceInfo]:
    index_number = None
    index_offset = None
    last_file_path = None
    channel_count = None
    sample_rate = None

    track_start = False  # if parser is between TRACK and INDEX commands
    last_title_file = None
    last_title_track = None

    tracks = []

    join = os.path.join

    for cmd, *args in cue_items:
        if cmd == CueCmd.TRACK or cmd == CueCmd.FILE or cmd == CueCmd.EOF:
            if index_number:
                assert last_title_track is not None
                assert index_offset is not None
                # noinspection PyTypeChecker
                tracks.append(TrackInfo(last_title_track, index_offset))
                index_number = None
            if cmd == CueCmd.TRACK:
                track_start = True
                continue

        if cmd == CueCmd.FILE or cmd == CueCmd.EOF:
            if last_file_path:
                yield AudioSourceInfo(
                    last_file_path,
                    last_title_file,
                    (),
                    channel_count,
                    sample_rate,
                    tracks)
                tracks = []
            if cmd == CueCmd.EOF:
                return

            last_file_path = join(directory_path, args[0])
            channel_count, sample_rate = _get_audio_properties(last_file_path)
        elif cmd == CueCmd.TITLE:
            if track_start:
                last_title_track = args[0]
            else:
                last_title_file = args[0]
        elif cmd == CueCmd.INDEX:
            track_start = False
            number, offset = args
            if not index_number or index_number > number:
                index_number, index_offset = number, int(sample_rate * offset)
        elif cmd == CueCmd.PERFORMER:
            continue  # TODO make use of performers
        else:
            raise NotImplementedError


def read_audio_info(in_path: str) -> Iterable[AudioSourceInfo]:
    """
    if input file is a cue, it can reference multiple audio files with different sample rates.
    therefore the result is a sequence.
    """
    kind = get_file_kind(in_path)
    if kind == FileKind.CUE:
        directory_path = os.path.dirname(in_path)
        return _translate_from_cue(directory_path, parse_cue(in_path))
    elif kind == FileKind.AUDIO:
        channel_count, sample_rate = _get_audio_properties(in_path)
        track_info = TrackInfo(name='', offset_samples=0)
        return [AudioSourceInfo(in_path, channel_count, sample_rate, [track_info])]
    else:
        raise NotImplementedError


def _get_audio_properties(in_path):
    channel_count, sample_rate = _get_params(in_path)
    if channel_count < 1 or sample_rate < 8000:
        sys.exit('invalid format: channels={}, sample_rate={}'.format(channel_count, sample_rate))

    return channel_count, sample_rate


def read_audio_data(what: AudioSourceInfo, samples_per_block: int) -> AudioSource:
    audio_blocks = _read_audio_blocks(what.path, what.channel_count, samples_per_block, what.tracks)
    return AudioSource(what, samples_per_block, audio_blocks)


def _test_ffmpeg():
    try:
        for n in (ex_ffmpeg, ex_ffprobe):
            sp.check_call((n, '-version'), stderr=DEVNULL, stdout=DEVNULL)
    except sp.CalledProcessError:
        sys.exit('ffmpeg not installed, broken or not on PATH')


def _parse_audio_params(s):
    d = {}
    for m in re.finditer(r'([a-z_]+)=([0-9]+)', s):
        v = m.groups()
        d.update({v[0]: int(v[1])})

    def values(channels, sample_rate):
        return channels, sample_rate

    return values(**d)


def _get_params(in_path):
    p = sp.Popen(
        (ex_ffprobe,
         '-v', 'error',
         '-select_streams', '0:a:0',
         '-show_entries', 'stream=channels,sample_rate',
         in_path),
        stdout=PIPE)
    out, err = p.communicate()
    returncode = p.returncode
    if returncode != 0:
        raise Exception('ffprobe returned {}'.format(returncode))
    out = out.decode('utf-8')
    return _parse_audio_params(out)


def _read_audio_blocks(in_path, channel_count, block_samples, tracks: List[TrackInfo]):
    bytes_per_sample = 4 * channel_count
    max_bytes_per_block = bytes_per_sample * block_samples

    p = sp.Popen(
        (ex_ffmpeg, '-v', 'error',
         '-i', in_path,
         '-map', '0:a:0',
         '-c:a', 'pcm_f32le',
         '-f', 'f32le',
         '-'),
        stderr=None,
        stdout=PIPE)

    sample_type = np.dtype('<f4')
    frombuffer = np.frombuffer
    reshape = np.reshape
    max_buffer = bytearray(max_bytes_per_block)
    with p.stdout as f:
        readinto = type(f).readinto

        skip_samples = tracks[0].offset_samples
        if skip_samples > 0:
            f.seek(bytes_per_sample * skip_samples, SEEK_CUR)


        def make_array(buffer, size):
            a = frombuffer(buffer, dtype=sample_type, count=size // 4)
            a = reshape(a, (channel_count, -1), order='F')
            return a


        def read_samples(n):
            """n = number of samples to read"""
            while n >= max_bytes_per_block:
                read_size = readinto(f, max_buffer)
                if read_size > 0:
                    yield make_array(max_buffer, read_size)
                else:
                    return
            if n > 0:
                tmp_buffer_size = n * bytes_per_sample
                tmp_buffer = bytearray(tmp_buffer_size)
                read_size = readinto(f, tmp_buffer)
                assert read_size == tmp_buffer_size
                yield make_array(tmp_buffer, read_size)

        # for ti, t in enumerate(tracks):
        #     track_offset = t.offset_samples


        # while True:
        #     read_size = readinto(f, max_buffer)
        #     if not read_size:
        #         break
        #
        #     a = frombuffer(max_buffer, dtype=sample_type, count=read_size // 4)
        #     a = reshape(a, (channel_count, -1), order='F')
        #     yield a
