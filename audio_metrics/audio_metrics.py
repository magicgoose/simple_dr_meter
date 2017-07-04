from typing import NamedTuple, Iterator

import numpy as np
from math import floor

from audio_io.audio_io import AudioSourceInfo


class DynamicRangeMetrics(NamedTuple):
    dr: int
    peak: float
    rms: float
    sample_count: int


def _calc_block_metrics(map_impl, samples: Iterator[np.ndarray], sample_count):
    def process_part(a: np.ndarray):
        length = a.shape[1]
        sample_count[0] += length

        a = np.ascontiguousarray(a)
        peaks = np.max(np.abs(a), axis=1)

        a = a ** 2  # numpy sometimes created readonly array so a**=2 won't always work
        sum_sqr = np.sum(a, axis=1)
        rms = np.sqrt(2.0 * sum_sqr / length)
        return peaks, rms, sum_sqr

    results = map_impl(process_part, samples)

    for peaks, rms, sum_sqr in results:
        yield from peaks
        yield from rms
        yield from sum_sqr


def decibel(a):
    return np.log10(a) * 20


def compute_dr(pool, a: AudioSourceInfo, samples: Iterator[np.ndarray]) -> DynamicRangeMetrics:
    channel_count = a.channel_count

    sample_count = [0]

    metrics = np.fromiter(_calc_block_metrics(pool, samples, sample_count), dtype='<f4').reshape((
        -1,  # number of block
        3,  # peak, rms, sum_sqr
        channel_count
    ))
    block_count = metrics.shape[0]
    sample_count = sample_count[0]

    peaks = metrics[:, 0, :]
    rms = metrics[:, 1, :]
    sum_of_squares_all = metrics[:, 2, :]

    peak_index = block_count - 2
    rms_percentile = 0.2

    total_second_peak = np.partition(peaks, peak_index, axis=0)[peak_index, :]

    rms_count = max(1, int(floor(block_count * rms_percentile)))

    rms_start = block_count - rms_count
    rms_range = range(rms_start, block_count)
    rms = np.partition(rms, rms_start, axis=0)[rms_range, :]
    rms **= 2
    rms_sqr_sum = np.sum(rms, axis=0)
    dr_per_channel = -decibel(np.sqrt(rms_sqr_sum / rms_count) / total_second_peak)

    dr = int(round(np.mean(dr_per_channel, axis=0)))

    peak_db = float(decibel(np.max(peaks)))

    sum_of_squares_all = np.sum(sum_of_squares_all, axis=0)
    rms_all = np.sqrt(2.0 * sum_of_squares_all / float(sample_count))
    rms_all = np.mean(rms_all)
    rms_db = float(decibel(rms_all))

    return DynamicRangeMetrics(dr, peak_db, rms_db, sample_count)
