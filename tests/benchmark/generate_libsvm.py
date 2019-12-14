"""Generate synthetic data in LibSVM format."""

import argparse
import io
import time

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

RNG = np.random.RandomState(2019)


def generate_data(args):
    """Generates the data."""
    print("Generating dataset: {} rows * {} columns".format(args.rows, args.columns))
    print("Sparsity {}".format(args.sparsity))
    print("{}/{} train/test split".format(1.0 - args.test_size, args.test_size))

    tmp = time.time()
    n_informative = args.columns * 7 // 10
    n_redundant = args.columns // 10
    n_repeated = args.columns // 10
    print("n_informative: {}, n_redundant: {}, n_repeated: {}".format(n_informative, n_redundant,
                                                                      n_repeated))
    x, y = make_classification(n_samples=args.rows, n_features=args.columns,
                               n_informative=n_informative, n_redundant=n_redundant,
                               n_repeated=n_repeated, shuffle=False, random_state=RNG)
    print("Generate Time: {} seconds".format(time.time() - tmp))

    tmp = time.time()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=args.test_size,
                                                        random_state=RNG, shuffle=False)
    print("Train/Test Split Time: {} seconds".format(time.time() - tmp))

    tmp = time.time()
    write_file('train.libsvm', x_train, y_train, args.sparsity)
    print("Write Train Time: {} seconds".format(time.time() - tmp))

    tmp = time.time()
    write_file('test.libsvm', x_test, y_test, args.sparsity)
    print("Write Test Time: {} seconds".format(time.time() - tmp))


def write_file(filename, x_data, y_data, sparsity):
    with open(filename, 'w') as f:
        for x, y in zip(x_data, y_data):
            write_line(f, x, y, sparsity)


def write_line(f, x, y, sparsity):
    with io.StringIO() as line:
        line.write(str(y))
        for i, col in enumerate(x):
            if 0.0 < sparsity < 1.0:
                if RNG.uniform(0, 1) > sparsity:
                    write_feature(line, i, col)
            else:
                write_feature(line, i, col)
        line.write('\n')
        f.write(line.getvalue())


def write_feature(line, index, feature):
    line.write(' ')
    line.write(str(index))
    line.write(':')
    line.write(str(feature))


def main():
    """The main function.

    Defines and parses command line arguments and calls the generator.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', type=int, default=1000000)
    parser.add_argument('--columns', type=int, default=50)
    parser.add_argument('--sparsity', type=float, default=0.0)
    parser.add_argument('--test_size', type=float, default=0.01)
    args = parser.parse_args()

    generate_data(args)


if __name__ == '__main__':
    main()
