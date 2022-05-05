from xgboost.sklearn import __model_doc, __estimator_doc, __objective_specific_doc
import sys
# sys.path = ["."] + sys.path

FIXES = {
    "typing.": "",
    ", NoneType": "",
    "TrainingCallback": '"TrainingCallback"'
}

def parse_doc_string(doc_string: str):
    params = {}
    for l in doc_string.splitlines():
        if l[:4] == ' ' * 4 and l[4] != ' ':
            param, param_type = l.lstrip().split(":")
            param = param.rstrip()
            param_type = param_type.lstrip()
            for base, fix in FIXES.items():
                param_type = param_type.replace(base, fix)
            if param_type.startswith("Optional["):
                assert param_type[-1] == "]"
                param_type = param_type[9:-1]
            params[param] = param_type
    return params

if __name__ == "__main__":
    params = {}
    for doc_string in [__model_doc, __estimator_doc, __objective_specific_doc]:
        params.update(parse_doc_string(doc_string))
    params.pop("kwargs", None)
    params_file_cont = [
        "import numpy",
        "from typing import Union, Callable, Tuple, List, Dict, TYPE_CHECKING",
        "from mypy_extensions import TypedDict",
        "",
        "if TYPE_CHECKING:",
        "    from .callback import TrainingCallback",
        "",
        "class Params(TypedDict, total=False):"
    ]
    for param_name, param_type in params.items():
        params_file_cont.append(f"    {param_name}: {param_type}")
    with open("xgboost/_parameter_typing.py", "w") as parameter_types:
        parameter_types.write("\n".join(params_file_cont))
