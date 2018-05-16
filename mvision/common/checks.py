"""
This file is borrowed directly from allennlp
"""

import logging

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """
    The exception is raised by any mv object when it's misconfigured
    """
    def __init__(self, message):
        super(ConfigurationError, self).__init__()
        self.message = message

    def __str__(self):
        return repr(self.message)


def log_pytorch_version_info():
    import torch
    logger.info("Pytorch version is: {}".format(torch.__version__))


def check_dimensions_match(dim_1: int,
                           dim_2: int,
                           dim_1_name: str,
                           dim_2_name: str) -> None:
    if dim_1 != dim_2:
        raise  ConfigurationError("{} must match {}, but got {} and {} instead".format(dim_1_name, dim_2_name,
                                                                                       dim_1, dim_2))

