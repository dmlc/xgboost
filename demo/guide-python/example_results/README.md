# 结果占位示例

本目录中的文件是格式说明，包含空值，并非真实训练结果。真实模型不能由占位文件替代；完成训练后生成的模型和评估文件位于本目录的上一级。

- `classification_metrics.example.json` 和 `regression_metrics.example.json` 展示性能文件的格式。
- 分类模型预期为 `../classification_model.json`，分类标签映射为 `../classification_classes.json`。
- 回归模型预期为 `../regression_model.json`。
- SHAP 结果预期为 `../shap_regression/` 和 `../shap_classification/class_0/` 等目录，每个目录生成 `shap_values.csv`、`feature_importance.csv`、`importance.png`、`summary.png` 和 `metadata.json`。
