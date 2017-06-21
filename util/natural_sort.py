import re

_num_pattern = re.compile('([0-9]+)')


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text
            for text in re.split(_num_pattern, s)]
