# this script is for preparation for PyPI installation package, 
# please don't use it for installing xgboost from github

# after executing `make pippack`, cd xgboost-python,
#run this script and get the sdist tar.gz in ./dist/
sh ./xgboost/build-python.sh
cp setup_pip.py setup.py
python setup.py sdist

#make sure you know what you gonna do, and uncomment the following line
#python setup.py register upload
