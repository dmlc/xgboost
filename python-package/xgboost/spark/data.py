import os
from typing import Iterator
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from xgboost import DMatrix


# Since sklearn's SVM converter doesn't address weights, this one does address weights:
def _dump_libsvm(features, labels, weights=None, external_storage_precision=5):
    esp = external_storage_precision
    lines = []

    def gen_label_str(row_idx):
        if weights is not None:
            return "{label:.{esp}g}:{weight:.{esp}g}".format(
                label=labels[row_idx], esp=esp, weight=weights[row_idx])
        else:
            return "{label:.{esp}g}".format(label=labels[row_idx], esp=esp)

    def gen_feature_value_str(feature_idx, feature_val):
        return "{idx:.{esp}g}:{value:.{esp}g}".format(
            idx=feature_idx, esp=esp, value=feature_val
        )

    is_csr_matrix = isinstance(features, csr_matrix)

    for i in range(len(labels)):
        current = [gen_label_str(i)]
        if is_csr_matrix:
            idx_start = features.indptr[i]
            idx_end = features.indptr[i + 1]
            for idx in range(idx_start, idx_end):
                j = features.indices[idx]
                val = features.data[idx]
                current.append(gen_feature_value_str(j, val))
        else:
            for j, val in enumerate(features[i]):
                current.append(gen_feature_value_str(j, val))
        lines.append(" ".join(current) + "\n")
    return lines


# This is the updated version that handles weights
def _stream_train_val_data(features, labels, weights, main_file,
                           external_storage_precision):
    lines = _dump_libsvm(features, labels, weights, external_storage_precision)
    main_file.writelines(lines)


def _stream_data_into_libsvm_file(data_iterator, has_weight,
                                  has_validation, file_prefix,
                                  external_storage_precision):
    # getting the file names for storage
    train_file_name = file_prefix + "/data.txt.train"
    train_file = open(train_file_name, "w")
    if has_validation:
        validation_file_name = file_prefix + "/data.txt.val"
        validation_file = open(validation_file_name, "w")

    train_val_data = _process_data_iter(data_iterator,
                                        train=True,
                                        has_weight=has_weight,
                                        has_validation=has_validation)
    if has_validation:
        train_X, train_y, train_w, _, val_X, val_y, val_w, _ = train_val_data
        _stream_train_val_data(train_X, train_y, train_w, train_file,
                               external_storage_precision)
        _stream_train_val_data(val_X, val_y, val_w, validation_file,
                               external_storage_precision)
    else:
        train_X, train_y, train_w, _ = train_val_data
        _stream_train_val_data(train_X, train_y, train_w, train_file,
                               external_storage_precision)

    if has_validation:
        train_file.close()
        validation_file.close()
        return train_file_name, validation_file_name
    else:
        train_file.close()
        return train_file_name


def _create_dmatrix_from_file(file_name, cache_name):
    if os.path.exists(cache_name):
        os.remove(cache_name)
    if os.path.exists(cache_name + ".row.page"):
        os.remove(cache_name + ".row.page")
    if os.path.exists(cache_name + ".sorted.col.page"):
        os.remove(cache_name + ".sorted.col.page")
    return DMatrix(file_name + "#" + cache_name)


def prepare_train_val_data(data_iterator,
                           has_weight,
                           has_validation,
                           has_fit_base_margin=False):
    def gen_data_pdf():
        for pdf in data_iterator:
            yield pdf

    return _process_data_iter(gen_data_pdf(),
                              train=True,
                              has_weight=has_weight,
                              has_validation=has_validation,
                              has_fit_base_margin=has_fit_base_margin,
                              has_predict_base_margin=False)


def prepare_predict_data(data_iterator, has_predict_base_margin):
    return _process_data_iter(data_iterator,
                              train=False,
                              has_weight=False,
                              has_validation=False,
                              has_fit_base_margin=False,
                              has_predict_base_margin=has_predict_base_margin)


def _check_feature_dims(num_dims, expected_dims):
    """
    Check all feature vectors has the same dimension
    """
    if expected_dims is None:
        return num_dims
    if num_dims != expected_dims:
        raise ValueError("Rows contain different feature dimensions: "
                         "Expecting {}, got {}.".format(
                             expected_dims, num_dims))
    return expected_dims


def _row_tuple_list_to_feature_matrix_y_w(data_iterator, train, has_weight,
                                          has_fit_base_margin,
                                          has_predict_base_margin,
                                          has_validation: bool = False):
    """
    Construct a feature matrix in ndarray format, label array y and weight array w
    from the row_tuple_list.
    If train == False, y and w will be None.
    If has_weight == False, w will be None.
    If has_base_margin == False, b_m will be None.
    Note: the row_tuple_list will be cleared during
    executing for reducing peak memory consumption
    """
    expected_feature_dims = None
    label_list, weight_list, base_margin_list = [], [], []
    label_val_list, weight_val_list, base_margin_val_list = [], [], []
    values_list, values_val_list = [], []

    # Process rows
    for pdf in data_iterator:
        if type(pdf) == tuple:
            pdf = pd.concat(list(pdf), axis=1, names=["values", "baseMargin"])

        if len(pdf) == 0:
            continue
        if train and has_validation:
            pdf_val = pdf.loc[pdf["validationIndicator"], :]
            pdf = pdf.loc[~pdf["validationIndicator"], :]

        num_feature_dims = len(pdf["values"].values[0])

        expected_feature_dims = _check_feature_dims(num_feature_dims,
                                                    expected_feature_dims)

        values_list.append(pdf["values"].to_list())
        if train:
            label_list.append(pdf["label"].to_list())
        if has_weight:
            weight_list.append(pdf["weight"].to_list())
        if has_fit_base_margin or has_predict_base_margin:
            base_margin_list.append(pdf.iloc[:, -1].to_list())
        if has_validation:
            values_val_list.append(pdf_val["values"].to_list())
            if train:
                label_val_list.append(pdf_val["label"].to_list())
            if has_weight:
                weight_val_list.append(pdf_val["weight"].to_list())
            if has_fit_base_margin or has_predict_base_margin:
                base_margin_val_list.append(pdf_val.iloc[:, -1].to_list())

    # Construct feature_matrix
    if expected_feature_dims is None:
        return [], [], [], []

    # Construct feature_matrix, y and w
    feature_matrix = np.concatenate(values_list)
    y = np.concatenate(label_list) if train else None
    w = np.concatenate(weight_list) if has_weight else None
    b_m = np.concatenate(base_margin_list) if (
            has_fit_base_margin or has_predict_base_margin) else None
    if has_validation:
        feature_matrix_val = np.concatenate(values_val_list)
        y_val = np.concatenate(label_val_list) if train else None
        w_val = np.concatenate(weight_val_list) if has_weight else None
        b_m_val = np.concatenate(base_margin_val_list) if (
                has_fit_base_margin or has_predict_base_margin) else None
        return feature_matrix, y, w, b_m, feature_matrix_val, y_val, w_val, b_m_val
    return feature_matrix, y, w, b_m


def _process_data_iter(data_iterator: Iterator[pd.DataFrame],
                       train: bool,
                       has_weight: bool,
                       has_validation: bool,
                       has_fit_base_margin: bool = False,
                       has_predict_base_margin: bool = False):
    """
    If input is for train and has_validation=True, it will split the train data into train dataset
    and validation dataset, and return (train_X, train_y, train_w, train_b_m <-
    train base margin, val_X, val_y, val_w, val_b_m <- validation base margin)
    otherwise return (X, y, w, b_m <- base margin)
    """
    if train and has_validation:
        train_X, train_y, train_w, train_b_m, val_X, val_y, val_w, val_b_m = \
            _row_tuple_list_to_feature_matrix_y_w(
                data_iterator, train, has_weight, has_fit_base_margin,
                has_predict_base_margin, has_validation)
        return train_X, train_y, train_w, train_b_m, val_X, val_y, val_w, val_b_m
    else:
        return _row_tuple_list_to_feature_matrix_y_w(data_iterator, train, has_weight,
                                                     has_fit_base_margin, has_predict_base_margin,
                                                     has_validation)


def convert_partition_data_to_dmatrix(partition_data_iter,
                                      has_weight,
                                      has_validation,
                                      use_external_storage=False,
                                      file_prefix=None,
                                      external_storage_precision=5):
    # if we are using external storage, we use a different approach for making the dmatrix
    if use_external_storage:
        if has_validation:
            train_file, validation_file = _stream_data_into_libsvm_file(
                partition_data_iter, has_weight,
                has_validation, file_prefix, external_storage_precision)
            training_dmatrix = _create_dmatrix_from_file(
                train_file, "{}/train.cache".format(file_prefix))
            val_dmatrix = _create_dmatrix_from_file(
                validation_file, "{}/val.cache".format(file_prefix))
            return training_dmatrix, val_dmatrix
        else:
            train_file = _stream_data_into_libsvm_file(
                partition_data_iter, has_weight,
                has_validation, file_prefix, external_storage_precision)
            training_dmatrix = _create_dmatrix_from_file(
                train_file, "{}/train.cache".format(file_prefix))
            return training_dmatrix

    # if we are not using external storage, we use the standard method of parsing data.
    train_val_data = prepare_train_val_data(partition_data_iter, has_weight, has_validation)
    if has_validation:
        train_X, train_y, train_w, _, val_X, val_y, val_w, _ = train_val_data
        training_dmatrix = DMatrix(data=train_X, label=train_y, weight=train_w)
        val_dmatrix = DMatrix(data=val_X, label=val_y, weight=val_w)
        return training_dmatrix, val_dmatrix
    else:
        train_X, train_y, train_w, _ = train_val_data
        training_dmatrix = DMatrix(data=train_X, label=train_y, weight=train_w)
        return training_dmatrix
