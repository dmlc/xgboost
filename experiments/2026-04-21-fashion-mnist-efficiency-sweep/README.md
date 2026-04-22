## Purpose

Sweep requested depth for a `fashion_mnist` lossguide model and compare efficiency error for:

- CPU TreeSHAP
- CPU QuadratureSHAP with `4` points
- CPU QuadratureSHAP with `6` points
- CPU QuadratureSHAP with `8` points
- CPU QuadratureSHAP with `16` points

The experiment records `mean`, `p99`, and `max` efficiency error, where efficiency is checked
against the raw margin:

`sum(phi) == predict(output_margin=True)`

The generated result directories are local experiment outputs and are not intended to be tracked.

## Commands

Original `max_leaves=128` run:

```bash
PYTHONPATH=/home/nfs/rorym/xgboost-wt/shapley-value-algorithms/python-package \
LD_LIBRARY_PATH=/home/nfs/rorym/xgboost-wt/shapley-value-algorithms/lib:${LD_LIBRARY_PATH} \
/home/nfs/rorym/anaconda3/bin/conda run -n xgboost python \
  /home/nfs/rorym/xgboost-wt/shapley-value-algorithms/experiments/2026-04-21-fashion-mnist-efficiency-sweep/benchmark_fashion_mnist_efficiency.py \
  --out-dir /home/nfs/rorym/xgboost-wt/shapley-value-algorithms/experiments/2026-04-21-fashion-mnist-efficiency-sweep/results \
  --points 4 8 16
```

Follow-up `max_leaves=1024` run:

```bash
PYTHONPATH=/home/nfs/rorym/xgboost-wt/shapley-value-algorithms/python-package \
LD_LIBRARY_PATH=/home/nfs/rorym/xgboost-wt/shapley-value-algorithms/lib:${LD_LIBRARY_PATH} \
/home/nfs/rorym/anaconda3/bin/conda run -n xgboost python \
  /home/nfs/rorym/xgboost-wt/shapley-value-algorithms/experiments/2026-04-21-fashion-mnist-efficiency-sweep/benchmark_fashion_mnist_efficiency.py \
  --out-dir /home/nfs/rorym/xgboost-wt/shapley-value-algorithms/experiments/2026-04-21-fashion-mnist-efficiency-sweep/results-maxleaves1024 \
  --max-leaves 1024 --depths 4 8 12 16 24 32 48 64 --points 4 6 8 16
```

## Generated Outputs

- `results.json`
- `results.csv`
- `summary.md`
- `efficiency_mean.png`
- `efficiency_p99.png`
- `efficiency_max.png`
