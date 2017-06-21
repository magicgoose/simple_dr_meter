#!/usr/bin/env python3
from datetime import datetime
import os
import sys

# import time
from typing import Iterable, Tuple, NamedTuple

from audio_io import read_audio_info, read_audio_data
from audio_io.audio_io import AudioSourceInfo
from audio_metrics import compute_dr


def get_samples_per_block(audio_info: AudioSourceInfo):
    sample_rate = audio_info.sample_rate
    sample_rate_extend = 60 if sample_rate == 44100 else sample_rate
    block_time = 3
    return block_time * (sample_rate + sample_rate_extend)


def get_log_path(in_path):
    if os.path.isdir(in_path):
        return in_path
    return os.path.dirname(in_path)


class LogGroup(NamedTuple):
    performers: Iterable[str]
    albums: Iterable[str]
    channels: int
    sample_rate: int
    tracks_dr: Iterable[Tuple[int, float, float, int, str]]


def get_group_title(group: LogGroup):
    return f'{", ".join(group.performers)} — {", ".join(group.albums)}'


def format_time(seconds):
    d = divmod
    m, s = d(seconds, 60)
    h, m = d(m, 60)
    if h:
        return f'{h}:{m:02d}:{s:02d}'
    return f'{m}:{s:02d}'


def write_log(out_path, dr_log_groups: Iterable[LogGroup], average_dr):
    out_path = os.path.join(out_path, 'dr.txt')
    print(f'writing log to {out_path}')
    with open(out_path, mode='x', encoding='utf8') as f:
        l1 = '-' * 80
        l2 = '=' * 80
        w = f.write
        w(f"log date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for group in dr_log_groups:
            group_name = get_group_title(group)

            w(f"{l1}\nAnalyzed: {group_name}\n{l1}\n\nDR         Peak         RMS     Duration Track\n{l1}\n")
            track_count = 0
            for dr, peak, rms, duration_sec, track_name in group.tracks_dr:
                w(f"DR{str(dr).ljust(4)}{peak:9.2f} dB{rms:9.2f} dB{format_time(duration_sec).rjust(10)} {track_name}\n")
                track_count += 1
            w(f"{l1}\n\nNumber of tracks:  {track_count}\nOfficial DR value: DR{average_dr}\n\n"
              f"Samplerate:        {group.sample_rate} Hz\nChannels:          {group.channels}\n{l2}\n\n")

    print('…done')


def flatmap(f, items):
    for i in items:
        yield from f(i)


def make_log_groups(l: Iterable[Tuple[AudioSourceInfo, Iterable[Tuple[int, float, float, int, str]]]]):
    import itertools
    grouped = itertools.groupby(l, key=lambda x: (x[0].channel_count, x[0].sample_rate))

    for ((channels, sample_rate), subitems) in grouped:
        subitems = tuple(subitems)
        performers = set(flatmap(lambda x: x[0].performers, subitems))
        albums = set(map(lambda x: x[0].album, subitems))
        tracks_dr = flatmap(lambda x: x[1], subitems)
        yield LogGroup(
            performers=performers,
            albums=albums,
            channels=channels,
            sample_rate=sample_rate,
            tracks_dr=tracks_dr)


def main():
    analyze_and_write_log()


def analyze_and_write_log():
    in_path = len(sys.argv) > 1 and sys.argv[1] or input()
    audio_info = read_audio_info(in_path)
    i = 0
    dr_mean = 0
    dr_log_items = []
    for audio_info_part in audio_info:
        dr_log_subitems = []
        dr_log_items.append((audio_info_part, dr_log_subitems))

        samples_per_block = get_samples_per_block(audio_info_part)
        audio_data = read_audio_data(audio_info_part, samples_per_block)
        for track_samples, track_info in zip(audio_data.blocks_generator, audio_info_part.tracks):
            dr_metrics = compute_dr(audio_info_part, track_samples)
            dr = dr_metrics.dr
            print(f"{(i+1):02d} - {track_info.name}: DR{dr}")
            dr_mean += dr
            duration_seconds = round(dr_metrics.sample_count / audio_info_part.sample_rate)
            dr_log_subitems.append(
                (dr, dr_metrics.peak, dr_metrics.rms, duration_seconds, f"{(i+1):02d}-{track_info.name}"))
            i += 1
    dr_mean /= i
    dr_mean = round(dr_mean)  # it's now official

    dr_log_items = make_log_groups(dr_log_items)
    write_log(get_log_path(in_path), dr_log_items, dr_mean)


if __name__ == '__main__':
    main()
