# type: ignore
"""Xgboost pyspark integration submodule for model API."""
# pylint: disable=fixme, invalid-name, protected-access, too-few-public-methods
import base64
import os
import uuid

from pyspark import SparkFiles, cloudpickle
from pyspark.ml.util import DefaultParamsReader, DefaultParamsWriter, MLReader, MLWriter
from pyspark.sql import SparkSession
from xgboost.core import Booster

from .utils import get_class_name, get_logger


def _get_or_create_tmp_dir():
    root_dir = SparkFiles.getRootDirectory()
    xgb_tmp_dir = os.path.join(root_dir, "xgboost-tmp")
    if not os.path.exists(xgb_tmp_dir):
        os.makedirs(xgb_tmp_dir)
    return xgb_tmp_dir


def deserialize_xgb_model(model_string, xgb_model_creator):
    """
    Deserialize an xgboost.XGBModel instance from the input model_string.
    """
    xgb_model = xgb_model_creator()
    xgb_model.load_model(bytearray(model_string.encode("utf-8")))
    return xgb_model


def serialize_booster(booster):
    """
    Serialize the input booster to a string.

    Parameters
    ----------
    booster:
        an xgboost.core.Booster instance
    """
    # TODO: change to use string io
    tmp_file_name = os.path.join(_get_or_create_tmp_dir(), f"{uuid.uuid4()}.json")
    booster.save_model(tmp_file_name)
    with open(tmp_file_name, encoding="utf-8") as f:
        ser_model_string = f.read()
    return ser_model_string


def deserialize_booster(ser_model_string):
    """
    Deserialize an xgboost.core.Booster from the input ser_model_string.
    """
    booster = Booster()
    # TODO: change to use string io
    tmp_file_name = os.path.join(_get_or_create_tmp_dir(), f"{uuid.uuid4()}.json")
    with open(tmp_file_name, "w", encoding="utf-8") as f:
        f.write(ser_model_string)
    booster.load_model(tmp_file_name)
    return booster


_INIT_BOOSTER_SAVE_PATH = "init_booster.json"


def _get_spark_session():
    return SparkSession.builder.getOrCreate()


class _SparkXGBSharedReadWrite:
    @staticmethod
    def saveMetadata(instance, path, sc, logger, extraMetadata=None):
        """
        Save the metadata of an xgboost.spark._SparkXGBEstimator or
        xgboost.spark._SparkXGBModel.
        """
        instance._validate_params()
        skipParams = ["callbacks", "xgb_model"]
        jsonParams = {}
        for p, v in instance._paramMap.items():  # pylint: disable=protected-access
            if p.name not in skipParams:
                jsonParams[p.name] = v

        extraMetadata = extraMetadata or {}
        callbacks = instance.getOrDefault(instance.callbacks)
        if callbacks is not None:
            logger.warning(
                "The callbacks parameter is saved using cloudpickle and it "
                "is not a fully self-contained format. It may fail to load "
                "with different versions of dependencies."
            )
            serialized_callbacks = base64.encodebytes(
                cloudpickle.dumps(callbacks)
            ).decode("ascii")
            extraMetadata["serialized_callbacks"] = serialized_callbacks
        init_booster = instance.getOrDefault(instance.xgb_model)
        if init_booster is not None:
            extraMetadata["init_booster"] = _INIT_BOOSTER_SAVE_PATH
        DefaultParamsWriter.saveMetadata(
            instance, path, sc, extraMetadata=extraMetadata, paramMap=jsonParams
        )
        if init_booster is not None:
            ser_init_booster = serialize_booster(init_booster)
            save_path = os.path.join(path, _INIT_BOOSTER_SAVE_PATH)
            _get_spark_session().createDataFrame(
                [(ser_init_booster,)], ["init_booster"]
            ).write.parquet(save_path)

    @staticmethod
    def loadMetadataAndInstance(pyspark_xgb_cls, path, sc, logger):
        """
        Load the metadata and the instance of an xgboost.spark._SparkXGBEstimator or
        xgboost.spark._SparkXGBModel.

        :return: a tuple of (metadata, instance)
        """
        metadata = DefaultParamsReader.loadMetadata(
            path, sc, expectedClassName=get_class_name(pyspark_xgb_cls)
        )
        pyspark_xgb = pyspark_xgb_cls()
        DefaultParamsReader.getAndSetParams(pyspark_xgb, metadata)

        if "serialized_callbacks" in metadata:
            serialized_callbacks = metadata["serialized_callbacks"]
            try:
                callbacks = cloudpickle.loads(
                    base64.decodebytes(serialized_callbacks.encode("ascii"))
                )
                pyspark_xgb.set(pyspark_xgb.callbacks, callbacks)
            except Exception as e:  # pylint: disable=W0703
                logger.warning(
                    f"Fails to load the callbacks param due to {e}. Please set the "
                    "callbacks param manually for the loaded estimator."
                )

        if "init_booster" in metadata:
            load_path = os.path.join(path, metadata["init_booster"])
            ser_init_booster = (
                _get_spark_session().read.parquet(load_path).collect()[0].init_booster
            )
            init_booster = deserialize_booster(ser_init_booster)
            pyspark_xgb.set(pyspark_xgb.xgb_model, init_booster)

        pyspark_xgb._resetUid(metadata["uid"])  # pylint: disable=protected-access
        return metadata, pyspark_xgb


class SparkXGBWriter(MLWriter):
    """
    Spark Xgboost estimator writer.
    """

    def __init__(self, instance):
        super().__init__()
        self.instance = instance
        self.logger = get_logger(self.__class__.__name__, level="WARN")

    def saveImpl(self, path):
        """
        save model.
        """
        _SparkXGBSharedReadWrite.saveMetadata(self.instance, path, self.sc, self.logger)


class SparkXGBReader(MLReader):
    """
    Spark Xgboost estimator reader.
    """

    def __init__(self, cls):
        super().__init__()
        self.cls = cls
        self.logger = get_logger(self.__class__.__name__, level="WARN")

    def load(self, path):
        """
        load model.
        """
        _, pyspark_xgb = _SparkXGBSharedReadWrite.loadMetadataAndInstance(
            self.cls, path, self.sc, self.logger
        )
        return pyspark_xgb


class SparkXGBModelWriter(MLWriter):
    """
    Spark Xgboost model writer.
    """

    def __init__(self, instance):
        super().__init__()
        self.instance = instance
        self.logger = get_logger(self.__class__.__name__, level="WARN")

    def saveImpl(self, path):
        """
        Save metadata and model for a :py:class:`_SparkXGBModel`
        - save metadata to path/metadata
        - save model to path/model.json
        """
        xgb_model = self.instance._xgb_sklearn_model
        _SparkXGBSharedReadWrite.saveMetadata(self.instance, path, self.sc, self.logger)
        model_save_path = os.path.join(path, "model")
        booster = xgb_model.get_booster().save_raw("json").decode("utf-8")
        _get_spark_session().sparkContext.parallelize([booster], 1).saveAsTextFile(
            model_save_path
        )


class SparkXGBModelReader(MLReader):
    """
    Spark Xgboost model reader.
    """

    def __init__(self, cls):
        super().__init__()
        self.cls = cls
        self.logger = get_logger(self.__class__.__name__, level="WARN")

    def load(self, path):
        """
        Load metadata and model for a :py:class:`_SparkXGBModel`

        :return: SparkXGBRegressorModel or SparkXGBClassifierModel instance
        """
        _, py_model = _SparkXGBSharedReadWrite.loadMetadataAndInstance(
            self.cls, path, self.sc, self.logger
        )

        xgb_sklearn_params = py_model._gen_xgb_params_dict(
            gen_xgb_sklearn_estimator_param=True
        )
        model_load_path = os.path.join(path, "model")

        ser_xgb_model = (
            _get_spark_session().sparkContext.textFile(model_load_path).collect()[0]
        )

        def create_xgb_model():
            return self.cls._xgb_cls()(**xgb_sklearn_params)

        xgb_model = deserialize_xgb_model(ser_xgb_model, create_xgb_model)
        py_model._xgb_sklearn_model = xgb_model
        return py_model
