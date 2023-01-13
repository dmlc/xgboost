from typing import Any, Dict

from xgboost._typing import FloatCompatible

from .base import _BuiltinObjFunction, objective_doc


@objective_doc
class LambdaMartNDCG(_BuiltinObjFunction):
    # fixme: maybe sequence of truncations?
    def __init__(self, truncation: FloatCompatible) -> None:
        self.truncation = truncation

    @staticmethod
    def name() -> str:
        return "lambdamart:ndcg"

    def _save_config(self) -> Dict[str, Any]:
        return {"lambdamart_param": {"ndcg_truncation": self.truncation}}
