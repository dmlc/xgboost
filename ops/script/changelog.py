"""Helper script for creating links to PRs for changelog. This should be used with the
`sphinx-issues` extension.

"""

import argparse
import os
import re

from test_utils import ROOT

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version",
        type=str,
        required=True,
        help="Major version of the changelog, e.g., 3.0.0 .",
    )
    args = parser.parse_args()
    version = args.version

    fname = os.path.join(ROOT, f"doc/changes/v{version}.rst")

    with open(fname) as fd:
        note = fd.read()

    # E.g. #11285 -> :pr:`11285`.
    regex = re.compile(r"(#)(\d+)")
    note = re.sub(regex, r":pr:`\2`", note)
    with open(fname, "w") as fd:
        fd.write(note)
