import sys
import re
import zipfile
import glob

if len(sys.argv) != 2:
    print('Usage: {} [wheel]'.format(sys.argv[0]))
    sys.exit(1)

vcomp140_path = 'C:\\Program Files (x86)\\Microsoft Visual Studio\\Shared\\14.0\\VC\\redist\\x64\\Microsoft.VC140.OpenMP\\vcomp140.dll'

for wheel_path in sorted(glob.glob(sys.argv[1])):
    m = re.search(r'xgboost-(.*)-py2.py3', wheel_path)
    assert m
    version = m.group(1)

    with zipfile.ZipFile(wheel_path, 'a') as f:
        f.write(vcomp140_path, 'xgboost-{}.data/data/xgboost/vcomp140.dll'.format(version))
