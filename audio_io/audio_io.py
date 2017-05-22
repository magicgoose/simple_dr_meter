import re
import subprocess as sp
from fractions import Fraction
from numbers import Number

import chardet

import sys
from subprocess import DEVNULL, PIPE
from typing import NamedTuple, Iterator, Sequence

import numpy as np
from os import path

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
    channel_count: int
    sample_rate: int
    tracks: Sequence[TrackInfo]


class AudioSource(NamedTuple):
    source_info: AudioSourceInfo
    samples_per_block: int
    blocks_generator: Iterator[np.ndarray]


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
        yield True


def read_audio_info(in_path: str) -> Sequence[AudioSourceInfo]:
    """
    if input file is a cue, it can reference multiple audio files with different sample rates.
    therefore the result is a sequence.
    """
    kind = get_file_kind(in_path)
    if kind == FileKind.CUE:
        for cue_cmd in parse_cue(in_path):
            print(cue_cmd)
        raise NotImplementedError
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
    audio_blocks = _read_audio_blocks(what.path, what.channel_count, samples_per_block)
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


def _read_audio_blocks(in_path, channel_count, block_samples):
    bytes_per_block = 4 * channel_count * block_samples
    p = sp.Popen(
        (ex_ffmpeg, '-v', 'error',
         '-i', in_path,
         '-map', '0:a:0',
         '-c:a', 'pcm_f32le',
         '-f', 'f32le',
         '-'),
        stderr=None,
        stdout=PIPE)

    with p.stdout as f:
        readinto = type(f).readinto
        buffer = bytearray(bytes_per_block)
        frombuffer = np.frombuffer
        reshape = np.reshape

        sample_type = np.dtype('<f4')

        while True:
            read_size = readinto(f, buffer)
            if not read_size:
                break

            a = frombuffer(buffer, dtype=sample_type, count=read_size // 4)
            a = reshape(a, (channel_count, -1), order='F')
            yield a
