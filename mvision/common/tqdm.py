"""
This file is borrowed from allennlp
"""

from tqdm import tqdm as _tqdm

_tqdm.monitor_interval = 0


class Tqdm:
    default_mininterval: float = 0.1

    @staticmethod
    def set_default_mininterval(value: float) -> None:
        Tqdm.default_mininterval = value

    @staticmethod
    def set_slower_interval(use_slower_interval: bool) -> None:
        if use_slower_interval:
            Tqdm.default_mininterval = 10.0
        else:
            Tqdm.default_mininterval = 0.1

    @staticmethod
    def tqdm(*args, **kwargs):
        new_kwargs = {
            'mininterval': Tqdm.default_mininterval,
            **kwargs
        }
        return _tqdm(*args, **new_kwargs)
