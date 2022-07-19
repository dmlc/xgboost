import os
from typing import Union


class DirectoryExcursion:
    def __init__(self, path: Union[os.PathLike, str]):
        self.path = path
        self.curdir = os.path.normpath(os.path.abspath(os.path.curdir))

    def __enter__(self):
        os.chdir(self.path)

    def __exit__(self, *args):
        os.chdir(self.curdir)
