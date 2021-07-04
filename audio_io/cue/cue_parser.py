import re
from enum import Enum, auto
from fractions import Fraction
from io import BufferedIOBase
from numbers import Number
from typing import Iterable

import chardet


class CueCmd(Enum):
    PERFORMER = auto()
    TITLE = auto()
    FILE = auto()
    TRACK = auto()
    INDEX = auto()
    REM = auto()
    EOF = auto()


def _unquote(s: str):
    return s[1 + s.index('"'):s.rindex('"')]


_whitespace_pattern = re.compile(r'\s+')
_rem_tag_pattern = re.compile(r'([A-Z_]+) (.+)')


def parse_cd_time(offset: str) -> Number:
    """parse time in CD-DA (75fps) format to seconds, exactly
    MM:SS:FF"""
    m, s, f = map(int, offset.split(':'))
    return m * 60 + s + Fraction(f, 75)


def _parse_cue_cmd(line: str, offset_in_seconds: bool = True):
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
        if offset_in_seconds:
            offset = parse_cd_time(offset)
        return CueCmd.INDEX, number, offset
    if cmd == 'REM':
        tag_name, tag_value = _rem_tag_pattern.fullmatch(args).groups()
        return CueCmd.REM, tag_name, tag_value

    return None


def read_cue_from_file(in_path: str) -> str:
    with open(in_path, 'rb') as f:
        assert isinstance(f, BufferedIOBase)
        content = f.read()
    encoding = chardet.detect(content)['encoding']
    return content.decode(encoding)


def parse_cue_str(content: str, offset_in_seconds: bool = True) -> Iterable[tuple]:
    for line in content.splitlines():
        cmd = _parse_cue_cmd(line, offset_in_seconds)
        if cmd:
            yield cmd
    yield CueCmd.EOF, None
