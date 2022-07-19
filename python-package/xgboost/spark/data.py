# type: ignore
"""Xgboost pyspark integration submodule for data related functions."""
# pylint: disable=too-many-arguments
from typing import Iterator

import numpy as np
import pandas as pd

from xgboost import DMatrix


def _prepare_train_val_data(
    data_iterator, has_weight, has_validation, has_fit_base_margin
):
    def gen_data_pdf():
        for pdf in data_iterator:
            yield pdf

    return _process_data_iter(
        gen_data_pdf(),
        train=True,
        has_weight=has_weight,
        has_validation=has_validation,
        has_fit_base_margin=has_fit_base_margin,
        has_predict_base_margin=False,
    )


def _check_feature_dims(num_dims, expected_dims):
    """
    Check all feature vectors has the same dimension
    """
    if expected_dims is None:
        return num_dims
    if num_dims != expected_dims:
        raise ValueError(
            f"Rows contain different feature dimensions: Expecting {expected_dims}, got {num_dims}."
        )
    return expected_dims


def _row_tuple_list_to_feature_matrix_y_w(
    data_iterator,
    train,
    has_weight,
    has_fit_base_margin,
    has_predict_base_margin,
    has_validation: bool = False,
):
    """
    Construct a feature matrix in ndarray format, label array y and weight array w
    from the row_tuple_list.
    If train == False, y and w will be None.
    If has_weight == False, w will be None.
    If has_base_margin == False, b_m will be None.
    Note: the row_tuple_list will be cleared during
    executing for reducing peak memory consumption
    """
    # pylint: disable=too-many-locals
    expected_feature_dims = None
    label_list, weight_list, base_margin_list = [], [], []
    label_val_list, weight_val_list, base_margin_val_list = [], [], []
    values_list, values_val_list = [], []

    # Process rows
    for pdf in data_iterator:
        if len(pdf) == 0:
            continue
        if train and has_validation:
            pdf_val = pdf.loc[pdf["validationIndicator"], :]
            pdf = pdf.loc[~pdf["validationIndicator"], :]

        num_feature_dims = len(pdf["values"].values[0])

        expected_feature_dims = _check_feature_dims(
            num_feature_dims, expected_feature_dims
        )

        # Note: each element in `pdf["values"]` is an numpy array.
        values_list.append(pdf["values"].to_list())
        if train:
            label_list.append(pdf["label"].to_numpy())
        if has_weight:
            weight_list.append(pdf["weight"].to_numpy())
        if has_fit_base_margin or has_predict_base_margin:
            base_margin_list.append(pdf["baseMargin"].to_numpy())
        if has_validation:
            values_val_list.append(pdf_val["values"].to_list())
            if train:
                label_val_list.append(pdf_val["label"].to_numpy())
            if has_weight:
                weight_val_list.append(pdf_val["weight"].to_numpy())
            if has_fit_base_margin or has_predict_base_margin:
                base_margin_val_list.append(pdf_val["baseMargin"].to_numpy())

    # Construct feature_matrix
    if expected_feature_dims is None:
        return [], [], [], []

    # Construct feature_matrix, y and w
    feature_matrix = np.concatenate(values_list)
    y = np.concatenate(label_list) if train else None
    w = np.concatenate(weight_list) if has_weight else None
    b_m = (
        np.concatenate(base_margin_list)
        if (has_fit_base_margin or has_predict_base_margin)
        else None
    )
    if has_validation:
        feature_matrix_val = np.concatenate(values_val_list)
        y_val = np.concatenate(label_val_list) if train else None
        w_val = np.concatenate(weight_val_list) if has_weight else None
        b_m_val = (
            np.concatenate(base_margin_val_list)
            if (has_fit_base_margin or has_predict_base_margin)
            else None
        )
        return feature_matrix, y, w, b_m, feature_matrix_val, y_val, w_val, b_m_val
    return feature_matrix, y, w, b_m


def _process_data_iter(
    data_iterator: Iterator[pd.DataFrame],
    train: bool,
    has_weight: bool,
    has_validation: bool,
    has_fit_base_margin: bool = False,
    has_predict_base_margin: bool = False,
):
    """
    If input is for train and has_validation=True, it will split the train data into train dataset
    and validation dataset, and return (train_X, train_y, train_w, train_b_m <-
    train base margin, val_X, val_y, val_w, val_b_m <- validation base margin)
    otherwise return (X, y, w, b_m <- base margin)
    """
    return _row_tuple_list_to_feature_matrix_y_w(
        data_iterator,
        train,
        has_weight,
        has_fit_base_margin,
        has_predict_base_margin,
        has_validation,
    )


def _convert_partition_data_to_dmatrix(
    partition_data_iter,
    has_weight,
    has_validation,
    has_base_margin,
    dmatrix_kwargs=None,
):
    # pylint: disable=too-many-locals, unbalanced-tuple-unpacking
    dmatrix_kwargs = dmatrix_kwargs or {}
    # if we are not using external storage, we use the standard method of parsing data.
    train_val_data = _prepare_train_val_data(
        partition_data_iter, has_weight, has_validation, has_base_margin
    )
    if has_validation:
        (
            train_x,
            train_y,
            train_w,
            train_b_m,
            val_x,
            val_y,
            val_w,
            val_b_m,
        ) = train_val_data
        training_dmatrix = DMatrix(
            data=train_x,
            label=train_y,
            weight=train_w,
            base_margin=train_b_m,
            **dmatrix_kwargs,
        )
        val_dmatrix = DMatrix(
            data=val_x,
            label=val_y,
            weight=val_w,
            base_margin=val_b_m,
            **dmatrix_kwargs,
        )
        return training_dmatrix, val_dmatrix

    train_x, train_y, train_w, train_b_m = train_val_data
    training_dmatrix = DMatrix(
        data=train_x,
        label=train_y,
        weight=train_w,
        base_margin=train_b_m,
        **dmatrix_kwargs,
    )
    return training_dmatrix
