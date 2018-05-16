"""
A logger that maintain logs of both stdout and stderr when models are run
"""

from typing import TextIO
import os

def repalce_cr_with_newline(message: str) -> str:
    if '\r' in message:
        message =message.replace('\r', '')
        if not message or message[-1] != '\n':
            message += '\n'
    return message

class TeeLogger:
    """
    This class is an attempt to maintain logs of both stdout and stderr for when models are run.
    To use this class, at the beginning of your script insert these lines::

        sys.stdout = TeeLogger("stdout.log", sys.stdout)
        sys.stderr = TeeLogger("stderr.log", sys.stderr)
    """
    def __init__(self, filename: str, terminal: TextIO, file_friendly_terminal_output: bool) -> None:
        self.terminal = terminal
        self.file_friendly_terminal_output = file_friendly_terminal_output
        parent_directory = os.path.dirname(filename)
        os.makedirs(parent_directory, exist_ok=True)
        self.log = open(filename, 'a')

    def write(self, message):
        cleaned = repalce_cr_with_newline(message)

        if self.file_friendly_terminal_output:
            self.terminal.write(cleaned)
        else:
            self.terminal.write(message)

        self.log.write(cleaned)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

