import sys

# import time

from audio_io import read_audio_info, read_audio_data
from audio_metrics import compute_dr


def main():
    in_path = len(sys.argv) > 1 and sys.argv[1] or input()

    audio_info = read_audio_info(in_path)

    for audio_info_part in audio_info:
        print(audio_info_part)
        # sample_rate = audio_info_part.sample_rate
        # sample_rate_extend = 60 if sample_rate == 44100 else sample_rate
        # block_time = 3
        # block_sample_count = block_time * (sample_rate + sample_rate_extend)
        #
        # audio_data = read_audio_data(audio_info_part, block_sample_count)
        #
        # dr = compute_dr(audio_data)
        # # t1 = time.perf_counter()
        # print(dr)
        # # t2 = time.perf_counter()
        # # print('Total processing time = {} seconds'.format(t2 - t1))


if __name__ == '__main__':
    main()
