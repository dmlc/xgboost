#*******************************************************************************
# Copyright 2017-2019 by Contributors
# \file bench_utils.py
# \brief utills for a benchmark for 'hist' tree_method on both CPU/GPU arhitectures
# \author Egor Smirnov
#*******************************************************************************

import os
import re
import bz2
import sys
import timeit
import tarfile
import requests
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

if sys.version_info[0] >= 3:
    from urllib.request import urlretrieve  # pylint: disable=import-error,no-name-in-module
else:
    from urllib import urlretrieve  # pylint: disable=import-error,no-name-in-module

DATASET_DIR="./data/"


def measure(func, string, nrepeat):
    t = timeit.Timer(stmt="%s()" % func.__name__, setup="from __main__ import %s" % func.__name__)
    res = t.repeat(repeat=nrepeat, number=1)

    def box_filter(timing, left=0.25, right=0.75): # statistically remove outliers and compute average
        timing.sort()
        size = len(timing)
        if size == 1:
            return timing[0]

        Q1, Q2 = timing[int(size * left)], timing[int(size * right)]

        IQ = Q2 - Q1

        lower = Q1 - 1.5 * IQ
        upper = Q2 + 1.5 * IQ

        result = np.array([item for item in timing if lower < item < upper])
        return np.mean(result)

    timing = box_filter(res)
    print((string + " = {:.4f} sec (").format(timing), res, ")")


def compute_logloss(y1, y2):
    return log_loss(y1.ravel(), y2)


def download_file(url):
    local_filename = DATASET_DIR + url.split('/')[-1]
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=2**20):
                if chunk:
                    f.write(chunk)
    return local_filename


def load_higgs(nrows_train, nrows_test, dtype):
    """
    Higgs dataset from UCI machine learning repository (
    https://archive.ics.uci.edu/ml/datasets/HIGGS).
    TaskType:binclass
    NumberOfFeatures:28
    NumberOfInstances:11M
    """
    if not os.path.isfile(DATASET_DIR + "HIGGS.csv.gz"):
        print("Loading data set...")
        download_file("https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz")

    print("Reading data set...")
    data = pd.read_csv(DATASET_DIR + "HIGGS.csv.gz", delimiter=",", header=None, compression="gzip", dtype=dtype, nrows=nrows_train+nrows_test)
    print("Pre-processing data set...")

    data = data[list(data.columns[1:])+list(data.columns[0:1])]
    n_features = data.shape[1]-1
    train_data = np.ascontiguousarray(data.values[:nrows_train,:n_features], dtype=dtype)
    train_label = np.ascontiguousarray(data.values[:nrows_train,n_features], dtype=dtype)
    test_data = np.ascontiguousarray(data.values[nrows_train:nrows_train+nrows_test,:n_features], dtype=dtype)
    test_label = np.ascontiguousarray(data.values[nrows_train:nrows_train+nrows_test,n_features], dtype=dtype)
    n_classes = len(np.unique(train_label))
    return train_data, train_label, test_data, test_label, n_classes


def load_higgs1m(dtype):
    return load_higgs(1000000, 500000, dtype)


def read_libsvm_msrank(file_obj, n_samples, n_features, dtype):
    X = np.zeros((n_samples, n_features))
    y = np.zeros((n_samples,))

    counter = 0

    regexp = re.compile(r'[A-Za-z0-9]+:(-?\d*\.?\d+)')

    for line in file_obj:
        line = str(line).replace("\\n'", "")
        line = regexp.sub('\g<1>', line)
        line = line.rstrip(" \n\r").split(' ')

        y[counter] = int(line[0])
        X[counter] = [float(i) for i in line[1:]]

        counter += 1
        if counter == n_samples:
            break

    return np.array(X, dtype=dtype), np.array(y, dtype=dtype)


def _make_gen(reader):
    b = reader(1024 * 1024)
    while b:
        yield b
        b = reader(1024 * 1024)


def _count_lines(filename):
    with open(filename, 'rb') as f:
        f_gen = _make_gen(f.read)
        return sum(buf.count(b'\n') for buf in f_gen)

def load_msrank_10k(dtype):
    """
    Dataset from szilard benchmarks: https://github.com/szilard/GBM-perf
    TaskType:binclass
    NumberOfFeatures:700
    NumberOfInstances:10100000
    """

    url = "https://storage.mds.yandex.net/get-devtools-opensource/471749/msrank.tar.gz"
    tar = DATASET_DIR + "msrank.tar.gz"

    if not os.path.isfile(tar):
        print("Loading data set...")
        download_file(url)

    if not os.path.isfile(DATASET_DIR + "MSRank/train.txt"):
        tar = tarfile.open(tar, "r:gz")
        tar.extractall(DATASET_DIR)
        tar.close()

    sets = []
    labels = []
    n_features = 137

    print("Reading data set...")
    for set_name in ['train.txt', 'vali.txt', 'test.txt']:
        file_name = DATASET_DIR + os.path.join('MSRank', set_name)

        n_samples = _count_lines(file_name)
        with open(file_name, 'r') as file_obj:
            X, y = read_libsvm_msrank(file_obj, n_samples, n_features, dtype)

        sets.append(X)
        labels.append(y)

    sets[0] = np.vstack((sets[0], sets[1]))
    labels[0] = np.hstack((labels[0], labels[1]))

    sets   = [ np.ascontiguousarray(sets[i]) for i in [0, 2]]
    labels = [ np.ascontiguousarray(labels[i]) for i in [0, 2]]

    n_classes = len(np.unique(labels[0]))

    return sets[0], labels[0], sets[1], labels[1], n_classes


def load_airline_one_hot(dtype):
    """
    Dataset from szilard benchmarks: https://github.com/szilard/GBM-perf
    TaskType:binclass
    NumberOfFeatures:700
    NumberOfInstances:10100000
    """
    url = 'https://s3.amazonaws.com/benchm-ml--main/'

    name_train = 'train-10m.csv'
    name_test = 'test.csv'

    sets = []
    labels = []

    categorical_names = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
    categorical_ids = [0, 1, 2, 4, 5, 6]

    numeric_names = ["DepTime", "Distance"]
    numeric_ids = [3, 7]

    for name in [name_train, name_test]:
        filename = os.path.join(DATASET_DIR, name)
        if not os.path.exists(filename):
            print("Loading", filename)
            urlretrieve(url + name, filename)

        print("Reading", filename)
        df = pd.read_csv(filename, nrows=1000000) if name == 'train-10m.csv' else pd.read_csv(filename)
        X = df.drop('dep_delayed_15min', 1)
        y = df["dep_delayed_15min"]

        y_num = np.where(y == "Y", 1, 0)

        sets.append(X)
        labels.append(y_num)

    n_samples_train = sets[0].shape[0]

    X = pd.concat(sets)
    X = pd.get_dummies(X, columns=categorical_names)
    sets = [X[:n_samples_train], X[n_samples_train:]]

    return sets[0], labels[0], sets[1], labels[1], 2
