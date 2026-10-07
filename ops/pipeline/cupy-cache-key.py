"""Fingerprint the GPU test environment without warming CuPy's kernel cache."""

import hashlib
import json
import os
import platform
import sys

import cupy
from cupy.cuda import nvrtc, runtime

metadata = {
    "architecture": platform.machine(),
    "python": list(sys.version_info[:3]),
    "cupy": cupy.__version__,
    "cuda_runtime": runtime.runtimeGetVersion(),
    "cuda_driver": runtime.driverGetVersion(),
    "nvrtc": nvrtc.getVersion(),
    "compute_capabilities": sorted(
        {
            cupy.cuda.Device(i).compute_capability
            for i in range(runtime.getDeviceCount())
        }
    ),
}
serialized = json.dumps(metadata, sort_keys=True)
print(f"CuPy cache environment: {serialized}")
fingerprint = hashlib.sha256(serialized.encode()).hexdigest()
with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
    output.write(f"fingerprint={fingerprint}\n")
