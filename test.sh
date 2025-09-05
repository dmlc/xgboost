export OMP_NUM_THREADS=1
pytest -vv -s --fulltrace tests/python/test_data_iterator.py::test_data_iterator
export OMP_NUM_THREADS=2
pytest -vv -s --fulltrace tests/python/test_data_iterator.py::test_data_iterator
export OMP_NUM_THREADS=4
pytest -vv -s --fulltrace tests/python/test_data_iterator.py::test_data_iterator
export OMP_NUM_THREADS=8
pytest -vv -s --fulltrace tests/python/test_data_iterator.py::test_data_iterator
export OMP_NUM_THREADS=16
pytest -vv -s --fulltrace tests/python/test_data_iterator.py::test_data_iterator
export OMP_NUM_THREADS=32
pytest -vv -s --fulltrace tests/python/test_data_iterator.py::test_data_iterator
export OMP_NUM_THREADS=64
pytest -vv -s --fulltrace tests/python/test_data_iterator.py::test_data_iterator
export OMP_NUM_THREADS=128
pytest -vv -s --fulltrace tests/python/test_data_iterator.py::test_data_iterator
export OMP_NUM_THREADS=224
pytest -vv -s --fulltrace tests/python/test_data_iterator.py::test_data_iterator
