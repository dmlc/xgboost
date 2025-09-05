cd _build
#rm -rf *
#cmake ..
make -j
cd ../python-package
pip install .
cd ..
