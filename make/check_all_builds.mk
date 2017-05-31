# simple makefile to verify all different ways to build xgboost
# currently tested only on linux!

default: makeOnlyCpu makeWithGpu cmakeOnlyCpu cmakeWithGpu

makeOnlyCpu:
	make clean_all
	make -j8
	make -j8 test
	make clean_all

makeWithGpu:
	make clean_all
	make PLUGIN_UPDATER_GPU=ON -j8
	make PLUGIN_UPDATER_GPU=ON -j8 test
	cd plugin/updater_gpu && PYTHONPATH=../../python-package python -m nose test/
	make clean_all

cmakeOnlyCpu:
	make clean_all
	mkdir build
	cd build && cmake .. && make -j8
	make clean_all

cmakeWithGpu:
	make clean_all
	mkdir build
	cd build && cmake .. -DPLUGIN_UPDATER_GPU=ON && make -j8
	cd plugin/updater_gpu && PYTHONPATH=../../python-package python -m nose test/
	make clean_all
