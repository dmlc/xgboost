"""Script to generate a TypedDict of Booster parameters from the doc-strings."""
from xgboost.sklearn import __model_doc, __estimator_doc, __objective_specific_doc

FIXES = {
    "typing.": "",
    ", NoneType": "",
    "TrainingCallback": '"TrainingCallback"',
    "FeatureTypes": '"FeatureTypes"'
}

def parse_doc_string(doc_string: str):
    """
    Parses parameters and typing information from the given doc-string.
    """
    parsed_params = {}
    for l in doc_string.splitlines():
        if l[:4] == ' ' * 4 and l[4] != ' ':
            parsed_name, parsed_type = l.lstrip().split(":")
            parsed_name = parsed_name.rstrip()
            parsed_type = parsed_type.lstrip()
            for base, fix in FIXES.items():
                parsed_type = parsed_type.replace(base, fix)
            parsed_params[parsed_name] = parsed_type
    return parsed_params

if __name__ == "__main__":
    params = {}
    for to_parse in [__model_doc, __estimator_doc, __objective_specific_doc]:
        params.update(parse_doc_string(to_parse))
    params.pop("kwargs", None)
    params_file_cont = [
        '"""Typing information for Booster parameters."""',
        "import numpy",
        "from typing import Union, Callable, Tuple, List, Dict, TYPE_CHECKING",
        "from mypy_extensions import TypedDict",
        "",
        "if TYPE_CHECKING:",
        "    from .callback import TrainingCallback",
        "    from ._typing import FeatureTypes",
        "",
        "class Params(TypedDict, total=False):"
    ]
    for param_name, param_type in params.items():
        params_file_cont.append(f"    {param_name}: {param_type}")
    with open("xgboost/_parameter_typing.py", "w", encoding="ascii") as parameter_types:
        parameter_types.write("\n".join(params_file_cont))
