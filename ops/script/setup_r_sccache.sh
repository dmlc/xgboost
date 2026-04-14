#!/bin/bash
# Configure R to use sccache for compiling packages.
# This script creates ~/.R/Makevars with sccache compiler wrappers.

set -euo pipefail

if [ -f ~/.R/Makevars ]; then
    echo "Error: ~/.R/Makevars already exists. Aborting to avoid overwriting."
    exit 1
fi

mkdir -p ~/.R
cat > ~/.R/Makevars << 'EOF'
CC = sccache gcc
CXX = sccache g++
CXX11 = sccache g++
CXX14 = sccache g++
CXX17 = sccache g++
CXX20 = sccache g++
EOF

echo "Configured R to use sccache via ~/.R/Makevars"
