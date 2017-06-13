#!/usr/bin/env bash
set -e

cat <<EOF
=====
Compile minimal libraries for XGBoost
=====

EOF

echo "Cleaning all..."
make clean_all > /dev/null

if [ "$(uname)" == "Linux" ]; then
    echo "Patching ...."
    (
        cd dmlc-core/
        git apply ../patches/*dmlc*
    )
fi

echo "Building..."
make config=make/minimum.mk
(
cd jvm-packages
./build_jars.sh
)

