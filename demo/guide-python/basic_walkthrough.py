"""Run the classification or regression CSV example selected by JSON."""

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "basic_walkthrough_config.json"


def load_config():
    try:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read {CONFIG_PATH}: {exc}") from exc
    if not isinstance(config, dict) or config.get("active_task") not in ("classification", "regression"):
        raise SystemExit("active_task must be classification or regression")
    task = config["active_task"]
    section = config.get(task)
    if not isinstance(section, dict) or not {"target_column", "num_boost_round", "params"}.issubset(section):
        raise SystemExit(f"{task} needs target_column, num_boost_round, and params")
    features = config.get("feature_columns")
    if not isinstance(features, list) or not features or not all(isinstance(x, str) and x for x in features) or len(features) != len(set(features)):
        raise SystemExit("feature_columns must be a nonempty list of unique column names")
    if section["target_column"] in features:
        raise SystemExit("target_column cannot appear in feature_columns")
    if not isinstance(section["params"], dict) or not isinstance(section["num_boost_round"], int) or section["num_boost_round"] < 1:
        raise SystemExit(f"{task} needs params as an object and a positive num_boost_round")
    for key in ("train_csv", "test_csv"):
        if not isinstance(config.get(key), str) or not config[key]:
            raise SystemExit(f"{key} must be a CSV path")
    return config, task, section


def read_samples(path, features, target):
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        raise SystemExit(f"Could not read CSV {path}: {exc}") from exc
    needed = features + [target]
    if frame.empty or not frame.columns.is_unique or not set(needed).issubset(frame.columns):
        raise SystemExit(f"CSV {path} must contain rows and columns: {', '.join(needed)}")
    if frame[needed].isna().any().any():
        raise SystemExit(f"CSV {path} has missing values in selected columns")
    try:
        frame[features] = frame[features].apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"CSV {path} has nonnumeric features: {exc}") from exc
    return frame


def main():
    config, task, section = load_config()
    features = config["feature_columns"]
    target = section["target_column"]
    train = read_samples(HERE / config["train_csv"], features, target)
    test = read_samples(HERE / config["test_csv"], features, target)
    if task == "classification":
        from classification_template import run
    else:
        from regression_template import run
    metrics = run(train, test, features, target, section, HERE)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
