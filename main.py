import sys

# import time

from audio_io import read_audio_info, read_audio_data
from audio_io.audio_io import AudioSourceInfo
from audio_metrics import compute_dr


def get_samples_per_block(audio_info: AudioSourceInfo):
    sample_rate = audio_info.sample_rate
    sample_rate_extend = 60 if sample_rate == 44100 else sample_rate
    block_time = 3
    return block_time * (sample_rate + sample_rate_extend)


def main():
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
    pass


if __name__ == '__main__':
    main()
