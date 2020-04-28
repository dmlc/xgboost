'''A demo for defining data iterator.

The demo that defines a customized iterator for passing batches of data into
`xgboost.DMatrix` and use this `DMatrix` for training.

Aftering going through the demo, one might ask why don't we use more native
Python iterator?  That's because XGBoost require a `reset` function, while
using `itertools.tee` might incur significant memory usage according to:

  https://docs.python.org/3/library/itertools.html#itertools.tee.

'''

import xgboost
import cupy
import numpy

COLS = 64
ROWS_PER_BATCH = 1000            # data is splited by rows
BATCHES = 32


class IterForDMatixDemo(xgboost.core.DataIter):
    '''A data iterator for XGBoost DMatrix.

    `reset` and `next` are required for any data iterator, other functions here
    are utilites for demonstration's purpose.

    '''
    def __init__(self):
        '''Generate some random data for demostration.

        Actual data can be anything that is currently supported by XGBoost.
        '''
        self.rows = ROWS_PER_BATCH
        self.cols = COLS
        rng = cupy.random.RandomState(1994)
        self._data = [rng.randn(self.rows, self.cols)] * BATCHES
        self._labels = [rng.randn(self.rows)] * BATCHES

        self.it = 0             # set iterator to 0
        super().__init__()

    def as_array(self):
        return cupy.concatenate(self._data)

    def as_array_labels(self):
        return cupy.concatenate(self._labels)

    def data(self):
        '''Utility function for obtaining current batch of data.'''
        return self._data[self.it]

    def labels(self):
        '''Utility function for obtaining current batch of label.'''
        return self._labels[self.it]

    def reset(self):
        '''Reset the iterator'''
        self.it = 0

    def next(self, input_data):
        '''Yield next batch of data'''
        if self.it == len(self._data):
            # Return 0 when there's no more batch.
            return 0
        input_data(data=self.data(), label=self.labels())
        self.it += 1
        return 1


def main():
    rounds = 100
    it = IterForDMatixDemo()

    # Use iterator
    m_with_it = xgboost.DMatrix(it)

    m = xgboost.DeviceQuantileDMatrix(
        it.as_array(), it.as_array_labels())

    assert m_with_it.num_col() == m.num_col()
    assert m_with_it.num_row() == m.num_row()

    reg_with_it = xgboost.train({'tree_method': 'gpu_hist'}, m_with_it,
                                num_boost_round=rounds)
    predict_with_it = reg_with_it.predict(m_with_it)

    reg = xgboost.train({'tree_method': 'gpu_hist'}, m,
                        num_boost_round=rounds)
    predict = reg.predict(m)

    numpy.testing.assert_allclose(predict_with_it, predict,
                                  rtol=1e6)


if __name__ == '__main__':
    main()
