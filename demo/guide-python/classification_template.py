"""Classification training template, called by basic_walkthrough.py."""

import json

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


def run(train, test, features, target, config, output_dir):
    import xgboost as xgb

    encoder = LabelEncoder()
    try:
        train_y = encoder.fit_transform(train[target])
        test_y = encoder.transform(test[target])
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"Classification labels are invalid or test CSV has unseen classes: {exc}") from exc
    count = len(encoder.classes_)
    if count < 2:
        raise SystemExit("Classification training requires at least two classes")
    params = dict(config["params"])
    params["objective"] = "binary:logistic" if count == 2 else "multi:softprob"
    params["eval_metric"] = "logloss" if count == 2 else "mlogloss"
    if count > 2:
        params["num_class"] = count
    else:
        params.pop("num_class", None)
    dtrain = xgb.DMatrix(train[features], label=train_y)
    dtest = xgb.DMatrix(test[features], label=test_y)
    model = xgb.train(params, dtrain, num_boost_round=config["num_boost_round"], evals=[(dtest, "test")])
    probabilities = model.predict(dtest)
    predicted = (probabilities >= 0.5).astype(int) if count == 2 else np.argmax(probabilities, axis=1)
    model.save_model(str(output_dir / "classification_model.json"))
    classes = [value.item() if hasattr(value, "item") else value for value in encoder.classes_]
    (output_dir / "classification_classes.json").write_text(json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics = {"task": "classification", "accuracy": float(accuracy_score(test_y, predicted)), "classes": classes}
    (output_dir / "classification_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics
