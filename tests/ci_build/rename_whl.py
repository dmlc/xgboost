import sys
import os
from contextlib import contextmanager

@contextmanager
def cd(path):
    path = os.path.normpath(path)
    cwd = os.getcwd()
    os.chdir(path)
    print("cd " + path)
    try:
        yield path
    finally:
        os.chdir(cwd)

if len(sys.argv) != 4:
    print('Usage: {} [wheel to rename] [commit id] [platform tag]'.format(sys.argv[0]))
    sys.exit(1)

whl_path = sys.argv[1]
commit_id = sys.argv[2]
platform_tag = sys.argv[3]

dirname, basename = os.path.dirname(whl_path), os.path.basename(whl_path)

with cd(dirname):
    tokens = basename.split('-')
    assert len(tokens) == 5
    keywords = {'pkg_name': tokens[0],
                'version': tokens[1],
                'commit_id': commit_id,
                'platform_tag': platform_tag}
    new_name = '{pkg_name}-{version}+{commit_id}-py3-none-{platform_tag}.whl'.format(**keywords)
    print('Renaming {} to {}...'.format(basename, new_name))
    os.rename(basename, new_name)
