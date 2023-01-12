import sys
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Type

from xgboost._typing import NumpyOrCupy
from xgboost.core import DMatrix, get_objective_info


class ObjFunction(ABC):
    """Base class for custom objective function."""

    @abstractmethod
    def get_gradient(
        self, y_predt: NumpyOrCupy, data: DMatrix
    ) -> Tuple[NumpyOrCupy, NumpyOrCupy]:
        raise NotImplementedError()

    @abstractmethod
    def pred_transform(self, y_predt: NumpyOrCupy) -> NumpyOrCupy:
        raise NotImplementedError()

    @abstractmethod
    def save_config(self) -> Dict[str, Any]:
        raise NotImplementedError()

    @abstractmethod
    def load_config(self, config: Dict[str, Any]) -> None:
        raise NotImplementedError()


class _BuiltinObjFunction:
    @staticmethod
    @abstractmethod
    def name() -> str:
        raise NotImplementedError()

    @abstractmethod
    def _save_config(self) -> Dict[str, Any]:
        return {}

    def save_config(self) -> Dict[str, Any]:
        config = self._save_config()
        config.update({"name": self.name()})
        return config


_DOC = None


def objective_doc(cls: Type) -> Type:
    global _DOC
    if _DOC is None:
        _DOC = get_objective_info()

    desc = _DOC[cls.name()]["desc"]
    doc = f"""{desc}"""
    arguments = _DOC[cls.name()].get("arguments", None)
    if not arguments:
        cls.__doc__ = doc
        _DOC.pop(cls.name())
        print("pop:", cls.name())
        return cls

    doc += """

    Parameters
    ----------

    """
    for arg in arguments:
        doc += arg["name"] + " :" + "\n"
        doc += "        " + arg["desc"] + "\n"
    cls.__doc__ = doc
    # We only need to initialize once.
    _DOC.pop(cls.name())
    print("pop:", cls.name())

    return cls


def make_obj(name: str, internal: str, mod_name: str) -> None:
    """Generate builtin objective."""
    objfn = type(name, (_BuiltinObjFunction,), {})
    setattr(objfn, "name", staticmethod(lambda: internal))
    objfn = objective_doc(objfn)
    mod = sys.modules[mod_name]
    assert not hasattr(mod, name), mod_name
    setattr(mod, name, objfn)


def debug_print_doc():
    import json
    print(json.dumps(_DOC, indent=2))
