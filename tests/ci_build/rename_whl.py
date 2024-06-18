import os
import sys

from test_utils import DirectoryExcursion

if len(sys.argv) != 4:
    print("Usage: {} [wheel to rename] [commit id] [platform tag]".format(sys.argv[0]))
    sys.exit(1)


whl_path = sys.argv[1]
commit_id = sys.argv[2]
platform_tag = sys.argv[3]

dirname, basename = os.path.dirname(whl_path), os.path.basename(whl_path)

with DirectoryExcursion(dirname):
    tokens = basename.split("-")
    assert len(tokens) == 5
    version = tokens[1].split("+")[0]
    keywords = {
        "pkg_name": tokens[0],
        "version": version,
        "commit_id": commit_id,
        "platform_tag": platform_tag,
    }
    new_name = "{pkg_name}-{version}+{commit_id}-py3-none-{platform_tag}.whl".format(
        **keywords
    )
    print("Renaming {} to {}...".format(basename, new_name))
    if os.path.isfile(new_name):
        os.remove(new_name)
    os.rename(basename, new_name)

    filesize = os.path.getsize(new_name) / 1024 / 1024  # MB
    print(f"Wheel size: {filesize}")

    msg = f"Limit of wheel size set by PyPI is exceeded. {new_name}: {filesize}"
    assert filesize <= 300, msg
