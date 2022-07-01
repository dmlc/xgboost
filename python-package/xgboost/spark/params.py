from pyspark.ml.param.shared import Param, Params


class HasArbitraryParamsDict(Params):
    """
    This is a Params based class that is extended by _XGBoostParams
    and holds the variable to store the **kwargs parts of the XGBoost
    input.
    """

    arbitraryParamsDict = Param(
        Params._dummy(),
        "arbitraryParamsDict",
        "This parameter holds all of the user defined parameters that"
        " the sklearn implementation of XGBoost can't recognize. "
        "It is stored as a dictionary.",
    )

    def setArbitraryParamsDict(self, value):
        return self._set(arbitraryParamsDict=value)

    def getArbitraryParamsDict(self, value):
        return self.getOrDefault(self.arbitraryParamsDict)


class HasBaseMarginCol(Params):
    """
    This is a Params based class that is extended by _XGBoostParams
    and holds the variable to store the base margin column part of XGboost.
    """

    baseMarginCol = Param(
        Params._dummy(),
        "baseMarginCol",
        "This stores the name for the column of the base margin",
    )

    def setBaseMarginCol(self, value):
        return self._set(baseMarginCol=value)

    def getBaseMarginCol(self, value):
        return self.getOrDefault(self.baseMarginCol)
