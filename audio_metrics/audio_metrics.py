from typing import NamedTuple

import numpy as np
from math import floor

from audio_io import AudioSource


class DynamicRangeMetrics(NamedTuple):
    overall_dr: int
    peak: np.ndarray
    rms: np.ndarray


def _calc_block_metrics(src: AudioSource):
    for a in src.blocks_generator:
        length = a.shape[1]

        a = np.ascontiguousarray(a)
        peaks = np.max(np.abs(a), axis=1)

        a **= 2  # original values are not needed after this point
        sum_sqr = np.sum(a, axis=1)

        rms = np.sqrt(2.0 * sum_sqr / length)
        yield from peaks
        yield from rms


def compute_dr(a: AudioSource) -> DynamicRangeMetrics:
    channel_count = a.source_info.channel_count
    metrics = np.fromiter(_calc_block_metrics(a), dtype='<f4').reshape((
        -1,  # number of block
        2,  # peak, rms
        channel_count
    ))
    block_count = metrics.shape[0]

    peaks = metrics[:, 0, :]
    rms = metrics[:, 1, :]

    peak_index = block_count - 2
    rms_percentile = 0.2

    total_second_peak = np.partition(peaks, peak_index, axis=0)[peak_index, :]

    rms_count = max(1, int(floor(block_count * rms_percentile)))

    rms_start = block_count - rms_count
    rms_range = range(rms_start, block_count)
    rms = np.partition(rms, rms_start, axis=0)[rms_range, :]
    rms **= 2
    rms_sqr_sum = np.sum(rms, axis=0)
    dr_per_channel = -20.0 * np.log10(np.sqrt(rms_sqr_sum / rms_count) / total_second_peak)

    dr = int(round(np.mean(dr_per_channel, axis=0)))

    # TODO: compute per-channel details (peak, rms)
    return DynamicRangeMetrics(dr, None, None)
