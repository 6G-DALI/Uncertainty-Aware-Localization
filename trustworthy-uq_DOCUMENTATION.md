# trustworthy-uq --- MLOps User Documentation

## 1. Installation

`trustworthy-uq` is a Python package for the 2D CSI-localization
uncertainty-quantification workflows used in the 6G-DALI work.

The current package is designed for Python 3.10+.

### 1.1 Recommended installation 
Navigate to the directory containing `pyproject.toml`:

``` bash
cd C:\path\to\trustworthy-uq
```

Install in editable mode:

``` bash
python -m pip install -e .
```

Editable installation is recommended during development because changes
to the package source are immediately visible without reinstalling.

For a normal, non-editable installation:

``` bash
python -m pip install .
```

### 1.2 Verify the installation

Run:

``` bash
python -c "import trustworthy_uq; print(trustworthy_uq.__version__)"
```

Expected output for the current release:

``` text
0.1.0
```

You can also check where Python loaded the package from:

``` bash
python -c "import trustworthy_uq; print(trustworthy_uq.__file__)"
```

### 1.3 Dependencies

The package declares these dependencies:

-   NumPy
-   pandas
-   PyTorch
-   Lightning
-   Plotly
-   Kaleido

They are installed automatically by:

``` bash
python -m pip install -e .
```

Kaleido is used for static Plotly image export. HTML plots can still be
produced without successful PNG export.

------------------------------------------------------------------------

# 2. What the package does

The package provides a ready-to-use workflow for adding uncertainty
information to an existing 2D CSI localization model.

The current supported localization backbone is **AttentionDenseNet
(ADN)**.

The package supports two uncertainty methods:

1.  **CQR --- one-sided multi-SLA conformalized prediction**
2.  **Adaptive SCP (aSCP) --- adaptive scale prediction followed by
    conformal calibration**

The package is intentionally focused on the current 2D localization use
case rather than being a fully generic uncertainty-quantification
framework.

The typical workflow is:

``` text
CSI data
   |
   v
ADN localization model
   |
   +----------------------------+
   |                            |
   v                            v
 CQR head                    aSCP scale model
   |                            |
   v                            v
Conformal calibration       Conformal calibration
   |                            |
   +-------------+--------------+
                 |
                 v
       Prediction + uncertainty radius
                 |
                 v
      SLA coverage / breach metrics
```

The ADN produces the point localization.

The UQ component produces a radius around that point. The radius is
associated with a requested target coverage/SLA level such as 90%, 95%,
or 99%.

------------------------------------------------------------------------

# 3. Important terminology for MLOps users

The package exposes a small number of concepts that are useful when
integrating it into an MLOps system.

### ADN

The existing localization model.

Input:

``` text
CSI sample
```

Output:

``` text
(x, y) localization
```

The ADN itself provides the point prediction.

### SLA level / target coverage

A requested probability of covering the true position.

Examples:

``` python
[0.90, 0.95, 0.99]
```

These correspond to:

``` text
90% target coverage
95% target coverage
99% target coverage
```

The package can process several SLA levels in one workflow.

### Radius

The uncertainty region is represented as a circle centered at the ADN
prediction.

For example:

``` text
ADN prediction
      +
      |
   radius
<---------->
```

A test sample is considered covered when its true position falls within
the predicted radius.

### Coverage

The fraction of test samples whose true location is inside the predicted
uncertainty region.

For example:

``` text
true_coverage = 0.956
```

means approximately 95.6% of the evaluated samples were covered.

### Breach rate

The fraction not covered:

``` text
breach_rate = 1 - true_coverage
```

For example:

``` text
coverage    = 0.956
breach rate = 0.044
```

------------------------------------------------------------------------

# 4. Package structure

The main package structure is:

``` text
trustworthy_uq/
|
+-- localization/
|   +-- adn.py
|   +-- backbone.py
|   +-- cqr.py
|   +-- adaptive_scp.py
|   +-- pipeline.py
|
+-- data/
|   +-- kul.py
|   +-- nomadic.py
|
+-- calibration/
|   +-- conformal.py
|
+-- evaluation/
|   +-- metrics.py
|   +-- visualization.py
|
+-- artifacts/
|   +-- io.py
|
+-- tracking/
|   +-- mlflow.py
|
+-- __init__.py
```

For normal MLOps integration, users generally only need:

``` python
from trustworthy_uq import LocalizationUQ
```

The lower-level modules are available when more specialized control is
required.

------------------------------------------------------------------------

# 5. Main entry point: LocalizationUQ

The main workflow class is:

``` python
from trustworthy_uq import LocalizationUQ
```

Create an instance:

``` python
uq = LocalizationUQ(
    sla_levels=[0.90, 0.95, 0.99],
    seed=42,
    batch_size=32,
    num_workers=0,
)
```

### Constructor parameters

``` python
LocalizationUQ(
    backbone=None,
    device=None,
    sla_levels=None,
    seed=42,
    batch_size=32,
    num_workers=0,
    split=(0.25, 0.45, 0.30),
)
```

  -----------------------------------------------------------------------
  Parameter                           Meaning
  ----------------------------------- -----------------------------------
  `backbone`                          Optional already-created ADN model

  `device`                            PyTorch device; automatically
                                      selects CUDA when available

  `sla_levels`                        SLA/coverage levels supported by
                                      the workflow

  `seed`                              Random seed used for dataset
                                      splitting

  `batch_size`                        Default inference/training batch
                                      size

  `num_workers`                       DataLoader workers

  `split`                             Dataset proportions for
                                      training/head, calibration, and
                                      test
  -----------------------------------------------------------------------

The default split is:

``` text
25% training/head
45% calibration
30% test
```

The split is performed by `split_dataset()`.

------------------------------------------------------------------------

# 6. Dataset handling

The current package supports the Nomadic/KUL CSI localization dataset
used by the experiments.

## 6.1 Single scenario

``` python
dataset = uq.build_scenario_dataset(
    data_dir="data/nomadic_dataset/ULA_lab_LoS",
    scenario_id=0,
)
```

The default configuration uses:

``` text
4 users
240 samples per user
```

These can be changed:

``` python
dataset = uq.build_scenario_dataset(
    data_dir=DATA_DIR,
    scenario_id=0,
    num_users=4,
    num_samples=240,
)
```

## 6.2 Multiple scenarios

``` python
dataset = uq.build_pooled_dataset(
    data_dir=DATA_DIR,
    scenario_ids=range(6),
)
```

This creates one combined dataset from the requested scenarios.

## 6.3 Dataset splitting

``` python
train_dataset, calibration_dataset, test_dataset = uq.split_dataset(dataset)
```

With the default split:

``` text
25% -> training/head
45% -> calibration
30% -> test
```

The same split must be used consistently when
training/calibrating/evaluating a workflow.

------------------------------------------------------------------------

# 7. ADN model workflows

ADN is the localization backbone.

## 7.1 Load an existing ADN model

``` python
uq.load_adn(
    "path/to/best_model.ckpt"
)
```

After this:

``` python
uq.backbone
```

contains the loaded model.

The package does not require the ADN checkpoint to have been trained by
`trustworthy-uq`.

Any compatible ADN checkpoint can be loaded.

## 7.2 Train a new ADN model

``` python
result = uq.train_adn(
    data_dir=DATA_DIR,
    output_dir="output/adn",
    batch_size=32,
    num_workers=0,
    num_users=4,
    num_samples=240,
    max_epochs=200,
    seed=42,
)
```

The function returns a dictionary containing the trained model and
checkpoint information.

Typical fields:

``` python
result["model"]
result["checkpoint_path"]
result["runtime_seconds"]
```

The best checkpoint is saved by Lightning under the specified output
directory.

Example:

``` text
output/
└── adn/
    └── checkpoints/
        └── best_adn_backbone.ckpt
```

The package does not require an external model registry.

The MLOps platform can copy/register the returned checkpoint path using
its own artifact/model-management mechanism.

------------------------------------------------------------------------

# 8. CQR workflow

CQR is the main multi-SLA uncertainty workflow.

The important operational point is:

**The CQR head is trained once for the configured SLA levels. It is not
retrained for every SLA calibration.**

For example:

``` python
SLA_LEVELS = [0.90, 0.95, 0.99]
```

creates one CQR head containing three SLA-specific radius outputs.

Calibration is then performed using the same trained head.

------------------------------------------------------------------------

# 9. Load an existing CQR head

First load the ADN:

``` python
uq.load_adn(ADN_CHECKPOINT)
```

Then load the CQR checkpoint:

``` python
uq.load_cqr(
    CQR_CHECKPOINT
)
```

The package obtains the configured SLA levels from the checkpoint.

------------------------------------------------------------------------

# 10. Train a new CQR head

A new CQR head can be trained on top of an already available ADN:

``` python
result = uq.train_cqr(
    train_dataset,
    validation_dataset=calibration_dataset,
    sla_levels=[0.90, 0.95, 0.99],
    max_epochs=100,
    lr=1e-3,
    weight_decay=1e-5,
    output_dir="output/cqr",
)
```

The backbone is frozen during CQR-head training.

The returned dictionary contains:

``` python
result["model"]
result["checkpoint_path"]
result["runtime_seconds"]
result["sla_levels"]
```

When `output_dir` is supplied, the best checkpoint is saved under:

``` text
output/cqr/
└── checkpoints/
    └── best_cqr_head.ckpt
```

------------------------------------------------------------------------

# 11. CQR calibration

After loading or training the CQR head:

``` python
calibration_result = uq.calibrate_cqr(
    calibration_dataset,
    sla_levels=[0.90, 0.95, 0.99],
    save_path="output/cqr/cqr_calibration.json",
)
```

Calibration determines the conformal adjustment needed for each
requested SLA.

The same trained CQR head can therefore be calibrated for:

``` python
[0.90]
```

or:

``` python
[0.90, 0.95, 0.99]
```

without retraining the head.

The calibration artifact contains the resulting conformal adjustment
values.

------------------------------------------------------------------------

# 12. aSCP workflow

Adaptive SCP is a second supported uncertainty workflow.

Operationally, it has two stages:

1.  Train an adaptive scale model on top of the frozen ADN.
2.  Apply conformal calibration to the normalized localization errors.

The adaptive scale model estimates how large the expected localization
error is for each input.

------------------------------------------------------------------------

# 13. Train a new aSCP model

``` python
result = uq.train_adaptive_scp(
    train_dataset,
    validation_dataset=calibration_dataset,
    max_epochs=60,
    lr=1e-3,
    weight_decay=1e-5,
    output_dir="output/ascp",
)
```

The ADN backbone is frozen.

The trained scale model is saved as:

``` text
output/ascp/
└── checkpoints/
    └── best_ascp_scale.ckpt
```

The returned result contains:

``` python
result["model"]
result["checkpoint_path"]
result["runtime_seconds"]
```

------------------------------------------------------------------------

# 14. Load an existing aSCP model

``` python
uq.load_ascp(
    "path/to/best_ascp_scale.ckpt"
)
```

This requires a compatible ADN backbone to already be loaded.

------------------------------------------------------------------------

# 15. aSCP calibration

After loading or training the scale model:

``` python
result = uq.calibrate_adaptive_scp(
    calibration_dataset,
    sla_levels=[0.90, 0.95, 0.99],
    save_path="output/ascp/ascp_calibration.json",
)
```

As with CQR, calibration can be performed for one or multiple SLA
levels.

The trained scale model itself is not retrained when changing the
requested SLA level.

------------------------------------------------------------------------

# 16. Single-sample prediction

For a raw CSI sample:

``` python
result = uq.predict(
    samples=X,
    sla_levels=[0.90, 0.95, 0.99],
    y_true=y,
    method="CQR",
)
```

The result contains:

``` python
result["method"]
result["sla_levels"]
result["point_prediction"]
result["radius_mm"]
result["error_mm"]
result["covered"]
```

For example:

``` text
point_prediction
    -> ADN x,y prediction

radius_mm
    -> uncertainty radius for each SLA

error_mm
    -> actual localization error, when y_true is supplied

covered
    -> whether the true position falls within each radius
```

The same API is used for aSCP:

``` python
result = uq.predict(
    samples=X,
    sla_levels=[0.90, 0.95, 0.99],
    y_true=y,
    method="aSCP",
)
```

------------------------------------------------------------------------

# 17. Test-set evaluation

For full test-set evaluation:

``` python
metrics, predictions = uq.evaluate(
    test_dataset,
    sla_levels=[0.90, 0.95, 0.99],
    method="CQR",
    output_dir="output/cqr_evaluation",
)
```

The first returned object is a compact metrics DataFrame.

The second is the per-sample prediction DataFrame.

## Main metrics

For CQR, the metrics include:

-   method
-   target assurance
-   alpha
-   calibration size
-   test size
-   conformal adjustment (`qhat_mm`)
-   mean localization error
-   median localization error
-   90th-percentile localization error
-   95th-percentile localization error
-   true coverage
-   breach rate
-   mean base radius
-   mean final radius

For aSCP, the corresponding radius statistics are reported as adaptive
interval-radius metrics.

------------------------------------------------------------------------

# 18. Understanding the CQR radius fields

CQR reports two radius values.

### Base radius

`mean_base_radius_mm`

This is the mean radius directly produced by the learned CQR head before
conformal calibration.

### Final radius

`mean_final_radius_mm`

This is the radius after adding the conformal calibration adjustment.

Conceptually:

``` text
final radius
    =
learned CQR radius
    +
conformal adjustment
```

The final radius is the one used to determine whether a sample is
covered.

Therefore, when assessing the deployed UQ behavior, use:

``` text
true_coverage
breach_rate
mean_final_radius_mm
```

rather than the base radius alone.

------------------------------------------------------------------------

# 19. Understanding aSCP radius fields

For aSCP, the learned model produces an adaptive scale for each sample.

The calibrated radius is conceptually:

``` text
final radius
    =
predicted scale
    ×
conformal calibration factor
```

This means that the radius can vary substantially between samples.

The same SLA can therefore result in different radii for different CSI
inputs.

------------------------------------------------------------------------

# 20. JSON artifacts

When `output_dir` is supplied to `evaluate()` or `run_experiment()`, the
package produces machine-readable JSON artifacts.

These are intended to make integration with an MLOps GUI
straightforward.

## 20.1 Whole-test-set result

``` text
uq_test_results.json
```

This is the main compact UQ result for an experiment.

It contains:

``` text
method
task
num_test_samples
model
calibration
sla_results
```

The `sla_results` section contains one entry for each requested SLA.

Typical information includes:

``` text
target assurance
alpha
test size
qhat
mean error
median error
P90 error
P95 error
true coverage
breach rate
mean radius
```

This file is the primary JSON artifact that an MLOps GUI should consume
for experiment-level UQ metrics.

------------------------------------------------------------------------

# 21. Single-sample JSON artifact

When single-sample analysis is enabled:

``` text
uq_single_sample.json
```

is generated.

It contains:

``` text
method
task
sample_index
model
point_prediction
true_position
sla_results
```

Each SLA entry contains information such as:

``` text
target coverage
radius
covered
localization error
base radius
conformal adjustment
```

This is suitable for a GUI panel showing:

``` text
Sample
  |
  +-- ADN prediction
  |
  +-- true location
  |
  +-- SLA 90% -> radius / covered
  +-- SLA 95% -> radius / covered
  +-- SLA 99% -> radius / covered
```

------------------------------------------------------------------------

# 22. Calibration JSON artifact

CQR:

``` text
cqr_calibration.json
```

aSCP:

``` text
ascp_calibration.json
```

These files contain the conformal calibration parameters.

They are useful for reproducing or loading calibration state.

They are not intended to replace `uq_test_results.json` as the main
experiment summary.

For an MLOps GUI, the recommended hierarchy is:

``` text
uq_test_results.json
    -> main experiment metrics

uq_single_sample.json
    -> individual prediction/UQ information

cqr_calibration.json
ascp_calibration.json
    -> calibration state
```

------------------------------------------------------------------------

# 23. CSV artifacts

Evaluation also creates detailed tabular artifacts.

For CQR:

``` text
cqr_metrics.csv
cqr_predictions.csv
```

For aSCP:

``` text
adaptive_scp_metrics.csv
adaptive_scp_predictions.csv
```

The metrics CSV contains one row per SLA level.

The predictions CSV contains one row per:

``` text
test sample × SLA level
```

It therefore contains substantially more information than the compact
JSON result.

For example, the prediction CSV contains:

``` text
pred_x
pred_y
true_x
true_y
euclidean_error_mm
base_radius_mm
qhat_mm
final_radius_mm
covered
```

These files are useful for detailed analysis or downstream
visualization.

------------------------------------------------------------------------

# 24. Plot artifacts

The package generates Plotly visualizations.

Typical outputs include:

``` text
cqr_coverage_vs_target.html
cqr_coverage_vs_target.png

cqr_mean_radius_vs_target.html
cqr_mean_radius_vs_target.png
```

For aSCP the prefix is:

``` text
adaptive_scp_
```

The HTML versions are particularly suitable for direct embedding in a
web-based MLOps interface.

The PNG files are useful for reports or static dashboards.

------------------------------------------------------------------------

# 25. Single-sample visualization

A single prediction can be visualized using:

``` python
uq.plot_prediction(
    prediction=[x_pred, y_pred],
    true_position=[x_true, y_true],
    radius=radius,
    sla_level=0.95,
    method="CQR",
    output_path="output/single_sample",
)
```

This produces a plot showing:

-   predicted position
-   true position
-   uncertainty circle

Use `plot_prediction()` when the prediction has already been computed.

Use `plot_sample()` when raw CSI data must first be passed through the
model.

------------------------------------------------------------------------

# 26. Recommended MLOps integration

The package does not need to contain an HTTP/API server.

A recommended architecture is:

``` text
MLOps GUI
    |
    v
MLOps API/service
    |
    v
trustworthy-uq
    |
    +--> dataset
    +--> ADN checkpoint
    +--> CQR/aSCP checkpoint
    |
    +--> calibration
    +--> inference
    +--> evaluation
    |
    v
Artifacts
```

The API layer can call the package and expose its outputs.

For example:

``` python
metrics, predictions = uq.evaluate(...)
```

The API can then return:

``` json
{
  "sla_results": {
    "0.95": {
      "true_coverage": 0.956,
      "breach_rate": 0.044,
      "mean_final_radius_mm": 22.80
    }
  }
}
```

Similarly, the API can return the contents of:

``` text
uq_single_sample.json
```

for a GUI single-prediction page.

There is therefore no need to add web-server functionality to the
package itself.

------------------------------------------------------------------------

# 27. Recommended API operations

A simple MLOps service can expose operations conceptually equivalent to:

### Load model

``` text
POST /uq/model/load
```

Inputs:

``` text
ADN checkpoint
CQR/aSCP checkpoint
method
```

### Calibrate

``` text
POST /uq/calibrate
```

Inputs:

``` text
calibration dataset
method
SLA levels
```

Output:

``` text
calibration JSON
```

### Evaluate

``` text
POST /uq/evaluate
```

Inputs:

``` text
test dataset
method
SLA levels
```

Output:

``` text
uq_test_results.json
```

### Single prediction

``` text
POST /uq/predict
```

Inputs:

``` text
CSI sample
method
SLA levels
```

Output:

``` text
prediction
radius for each SLA
```

### Single-sample visualization

``` text
GET /uq/sample/{id}/plot
```

The actual endpoint naming is outside the package and can be chosen by
the MLOps team.

------------------------------------------------------------------------

# 28. One-call experiment workflow

For reproducing the complete experiment workflow, use:

``` python
uq.run_experiment(...)
```

This is the easiest function for an experiment runner or MLOps
orchestration layer.

Example using existing models:

``` python
uq.run_experiment(
    data_dir=DATA_DIR,
    method="CQR",
    eval_mode="pooled",
    scenario_ids=range(6),
    adn_checkpoint=ADN_CHECKPOINT,
    cqr_checkpoint=CQR_CHECKPOINT,
    train_new_adn=False,
    train_new_cqr=False,
    train_new_ascp=False,
    sla_levels=[0.90, 0.95, 0.99],
    output_dir="output/experiment",
    enable_single_sample=True,
    single_sample_index=0,
    single_sample_sla=0.95,
)
```

For aSCP:

``` python
uq.run_experiment(
    data_dir=DATA_DIR,
    method="aSCP",
    eval_mode="pooled",
    scenario_ids=range(6),
    adn_checkpoint=ADN_CHECKPOINT,
    ascp_checkpoint=ASCP_CHECKPOINT,
    train_new_adn=False,
    train_new_ascp=False,
    sla_levels=[0.90, 0.95, 0.99],
    output_dir="output/experiment",
    enable_single_sample=True,
)
```

------------------------------------------------------------------------

# 29. Training from scratch through run_experiment

The one-call workflow can also train models.

### New CQR head

``` python
uq.run_experiment(
    data_dir=DATA_DIR,
    method="CQR",
    adn_checkpoint=ADN_CHECKPOINT,
    train_new_cqr=True,
    sla_levels=[0.90, 0.95, 0.99],
    output_dir="output/new_cqr",
)
```

The new CQR checkpoint is saved below the experiment output directory.

### New aSCP model

``` python
uq.run_experiment(
    data_dir=DATA_DIR,
    method="aSCP",
    adn_checkpoint=ADN_CHECKPOINT,
    train_new_ascp=True,
    sla_levels=[0.90, 0.95, 0.99],
    output_dir="output/new_ascp",
)
```

The new aSCP checkpoint is saved below the experiment output directory.

### New ADN

The workflow can also train an ADN:

``` python
uq.run_experiment(
    data_dir=DATA_DIR,
    method="CQR",
    train_new_adn=True,
    train_new_cqr=True,
    sla_levels=[0.90, 0.95, 0.99],
    output_dir="output/full_training",
)
```

The package returns the trained models/checkpoint paths through the
underlying training functions, while the MLOps system can register or
copy those checkpoints as required.

------------------------------------------------------------------------

# 30. Model artifact management

The package does not require a model registry.

For MLOps integration, the recommended approach is:

``` text
Model registry / artifact store
            |
            v
        checkpoint
            |
            v
trustworthy-uq.load_adn()
trustworthy-uq.load_cqr()
trustworthy-uq.load_ascp()
```

For newly trained models:

``` text
trustworthy-uq training
        |
        v
checkpoint file
        |
        v
MLOps artifact/model registry
```

This keeps model lifecycle management separate from the UQ package.

------------------------------------------------------------------------

# 31. MLflow integration

The package contains a small optional MLflow adapter:

``` python
from trustworthy_uq.tracking.mlflow import log_evaluation
```

It can log:

-   metrics CSV
-   predictions CSV
-   generated figures

Example:

``` python
log_evaluation(
    mlflow,
    metrics_df=metrics,
    metrics_csv=Path("output/cqr/cqr_metrics.csv"),
    predictions_csv=Path("output/cqr/cqr_predictions.csv"),
    figures_dir=Path("output/cqr"),
)
```

The package does not require MLflow for its core operation.

An MLOps environment can therefore use:

-   MLflow
-   another experiment tracker
-   a custom artifact store
-   direct filesystem/object-storage APIs

without changing the UQ computation itself.

------------------------------------------------------------------------

# 32. Recommended production workflow

For an existing trained ADN and existing CQR model:

``` python
from trustworthy_uq import LocalizationUQ

uq = LocalizationUQ(
    sla_levels=[0.90, 0.95, 0.99]
)

uq.load_adn(ADN_CHECKPOINT)

uq.load_cqr(CQR_CHECKPOINT)

dataset = uq.build_pooled_dataset(
    DATA_DIR,
    scenario_ids=range(6)
)

train, calibration, test = uq.split_dataset(dataset)

uq.calibrate_cqr(
    calibration,
    sla_levels=[0.90, 0.95, 0.99],
    save_path="output/cqr_calibration.json"
)

metrics, predictions = uq.evaluate(
    test,
    sla_levels=[0.90, 0.95, 0.99],
    method="CQR",
    output_dir="output/cqr"
)
```

For most MLOps applications, however, `run_experiment()` is simpler
because it combines these steps.

------------------------------------------------------------------------

# 33. What an MLOps developer needs to provide

For a standard inference/evaluation workflow, the MLOps system needs:

### Required

1.  CSI dataset
2.  compatible ADN checkpoint
3.  compatible CQR or aSCP checkpoint
4.  calibration data
5.  requested SLA levels

### Optional

6.  test dataset
7.  output/artifact directory
8.  MLflow or other tracking system
9.  GUI/API layer

The MLOps developer does not need to implement the conformal calibration
mathematics.

------------------------------------------------------------------------

# 34. What the package returns versus what it saves

There are two mechanisms.

### Python return values

Functions such as:

``` python
train_cqr()
train_adaptive_scp()
calibrate_cqr()
calibrate_adaptive_scp()
predict()
evaluate()
run_experiment()
```

return Python objects/dictionaries/DataFrames.

### Filesystem artifacts

When an output directory or save path is supplied, the package can save:

``` text
model checkpoints
calibration JSON
test-result JSON
single-sample JSON
metrics CSV
prediction CSV
HTML plots
PNG plots
```

This separation is useful for MLOps because the API can consume returned
objects directly or retrieve persisted artifacts.

------------------------------------------------------------------------

# 35. Error handling and prerequisites

The package intentionally keeps exception handling relatively
lightweight.

Before calling the UQ functions, ensure:

1.  The ADN model is loaded.
2.  The required CQR/aSCP model is loaded or trained.
3.  Calibration has been performed before `predict()` or `evaluate()`.
4.  Requested SLA levels exist in the trained CQR model.
5.  The input CSI tensor has the expected shape.
6.  The dataset uses the expected Nomadic/KUL format.

A common sequence is:

``` text
load ADN
   ↓
load/train CQR or aSCP
   ↓
calibrate
   ↓
predict/evaluate
```

------------------------------------------------------------------------

# 36. Common issues

## `ModuleNotFoundError: trustworthy_uq`

Usually the IDE is using a different Python interpreter.

Check:

``` python
import sys
print(sys.executable)
```

Then:

``` python
import trustworthy_uq
print(trustworthy_uq.__file__)
```

If necessary, reinstall into that interpreter:

``` bash
python -m pip install -e .
```

## Lightning import errors

The package requires a compatible Lightning installation.

Use:

``` bash
python -m pip install -U lightning
```

and verify:

``` python
import lightning
print(lightning.__version__)
```

## Missing checkpoint

Verify the path before loading:

``` python
from pathlib import Path

path = Path("path/to/checkpoint.ckpt")
print(path.exists())
```

## `KeyError` for an SLA level

For CQR, the requested SLA must be one of the SLA levels with which the
CQR head was trained.

For example, if the head was trained for:

``` python
[0.90, 0.95, 0.99]
```

requesting:

``` python
[0.975]
```

is not supported by that checkpoint.

Train a CQR head configured with the required SLA levels if that SLA is
needed.

## Prediction before calibration

`predict()` requires conformal calibration values.

The usual sequence is:

``` python
uq.calibrate_cqr(...)
uq.predict(...)
```

or:

``` python
uq.calibrate_adaptive_scp(...)
uq.predict(..., method="aSCP")
```

------------------------------------------------------------------------

# 37. Minimal CQR example

``` python
from trustworthy_uq import LocalizationUQ

DATA_DIR = "data/nomadic_dataset/ULA_lab_LoS"

uq = LocalizationUQ(
    sla_levels=[0.90, 0.95, 0.99]
)

uq.load_adn("best_model.ckpt")
uq.load_cqr("best_cqr_head.ckpt")

dataset = uq.build_pooled_dataset(
    DATA_DIR,
    scenario_ids=range(6)
)

train, calibration, test = uq.split_dataset(dataset)

uq.calibrate_cqr(
    calibration,
    sla_levels=[0.90, 0.95, 0.99],
    save_path="output/cqr_calibration.json"
)

metrics, predictions = uq.evaluate(
    test,
    sla_levels=[0.90, 0.95, 0.99],
    method="CQR",
    output_dir="output/cqr"
)

print(metrics)
```

------------------------------------------------------------------------

# 38. Minimal aSCP example

``` python
from trustworthy_uq import LocalizationUQ

DATA_DIR = "data/nomadic_dataset/ULA_lab_LoS"

uq = LocalizationUQ(
    sla_levels=[0.90, 0.95, 0.99]
)

uq.load_adn("best_model.ckpt")
uq.load_ascp("best_ascp_scale.ckpt")

dataset = uq.build_pooled_dataset(
    DATA_DIR,
    scenario_ids=range(6)
)

train, calibration, test = uq.split_dataset(dataset)

uq.calibrate_adaptive_scp(
    calibration,
    sla_levels=[0.90, 0.95, 0.99],
    save_path="output/ascp_calibration.json"
)

metrics, predictions = uq.evaluate(
    test,
    sla_levels=[0.90, 0.95, 0.99],
    method="aSCP",
    output_dir="output/ascp"
)

print(metrics)
```

------------------------------------------------------------------------

# 39. Minimal MLOps-oriented workflow

For an MLOps service, the core sequence can be reduced to:

``` python
uq = LocalizationUQ(
    sla_levels=[0.90, 0.95, 0.99]
)

uq.load_adn(adn_checkpoint)

uq.load_cqr(cqr_checkpoint)

metrics, predictions = uq.run_experiment(
    data_dir=data_dir,
    method="CQR",
    eval_mode="pooled",
    scenario_ids=range(6),
    adn_checkpoint=adn_checkpoint,
    cqr_checkpoint=cqr_checkpoint,
    sla_levels=[0.90, 0.95, 0.99],
    output_dir=output_dir,
    enable_single_sample=True,
)
```

The resulting output directory contains the main machine-readable
artifacts that can be exposed through the MLOps API.

------------------------------------------------------------------------

# 40. API reference

## `LocalizationUQ`

### Data

``` python
build_scenario_dataset(
    data_dir,
    scenario_id,
    num_users=4,
    num_samples=240
)
```

``` python
build_pooled_dataset(
    data_dir,
    scenario_ids=range(6),
    num_users=4,
    num_samples=240
)
```

``` python
split_dataset(dataset)
```

### ADN

``` python
load_adn(checkpoint_path)
```

``` python
train_adn(data_dir, **kwargs)
```

### CQR

``` python
train_cqr(
    train_dataset,
    validation_dataset=None,
    sla_levels=None,
    max_epochs=100,
    lr=1e-3,
    weight_decay=1e-5,
    output_dir=None
)
```

``` python
load_cqr(checkpoint_path)
```

``` python
calibrate_cqr(
    calibration_dataset,
    sla_levels=None,
    save_path=None
)
```

### aSCP

``` python
train_adaptive_scp(
    train_dataset,
    validation_dataset=None,
    max_epochs=60,
    lr=1e-3,
    weight_decay=1e-5,
    output_dir=None
)
```

``` python
load_ascp(checkpoint_path)
```

``` python
calibrate_adaptive_scp(
    calibration_dataset,
    sla_levels=None,
    save_path=None
)
```

### Inference

``` python
predict(
    samples,
    sla_levels=None,
    y_true=None,
    method="CQR",
    output_path=None
)
```

### Evaluation

``` python
evaluate(
    test_dataset,
    sla_levels=None,
    method="CQR",
    output_dir=None
)
```

### Visualization

``` python
plot_sample(
    samples,
    y_true,
    sample_index=0,
    sla_level=0.95,
    method="CQR",
    output_path=None
)
```

``` python
plot_prediction(
    prediction,
    true_position,
    radius,
    sla_level=0.95,
    method="CQR",
    output_path=None
)
```

### Complete workflow

``` python
run_experiment(
    data_dir,
    method="CQR",
    eval_mode="pooled",
    scenario_id=0,
    scenario_ids=range(6),
    adn_checkpoint=None,
    cqr_checkpoint=None,
    ascp_checkpoint=None,
    train_new_adn=False,
    train_new_cqr=False,
    train_new_ascp=False,
    sla_levels=None,
    output_dir="output/unified_conformal_experiments",
    enable_single_sample=True,
    single_sample_index=0,
    single_sample_sla=0.95,
    **train_kwargs
)
```

------------------------------------------------------------------------

# 41. Recommended artifact contract for the MLOps GUI

For integration, the simplest contract is:

``` text
Input
-----
dataset
ADN checkpoint
UQ checkpoint
method
SLA levels


Output
------
uq_test_results.json
uq_single_sample.json
cqr_calibration.json / ascp_calibration.json
metrics CSV
predictions CSV
HTML/PNG plots
```

The GUI does not need to understand the internals of conformal
prediction.

It can treat the package as a service that returns:

``` text
point prediction
+
uncertainty radius
+
target SLA
+
empirical coverage
+
breach rate
+
radius statistics
```

This is the intended abstraction boundary between the UQ implementation
and the MLOps layer.

------------------------------------------------------------------------

# 42. Current scope

The current release is intentionally focused on:

-   2D CSI localization
-   Nomadic/KUL dataset workflow
-   ADN localization backbone
-   one-sided multi-SLA CQR
-   adaptive SCP with a learned scale model
-   conformal calibration
-   test-set coverage and breach evaluation
-   single-sample UQ
-   machine-readable artifacts
-   optional MLflow artifact logging

It is not currently intended as a generic framework for arbitrary
regression models, arbitrary datasets, or arbitrary UQ algorithms.

The package structure, however, is designed so additional UQ methods and
classification workflows can be added later.

------------------------------------------------------------------------

# 43. Operational summary

For an MLOps user, the complete lifecycle is:

``` text
1. Install package
       ↓
2. Load ADN checkpoint
       ↓
3. Load or train CQR/aSCP model
       ↓
4. Build/select dataset
       ↓
5. Calibrate using calibration data
       ↓
6. Run prediction/evaluation
       ↓
7. Retrieve JSON/CSV/plots
       ↓
8. Expose results through MLOps API/GUI
```

The MLOps layer is responsible for:

-   model/data artifact storage
-   authentication and API endpoints
-   experiment tracking
-   GUI presentation
-   model lifecycle management

`trustworthy-uq` is responsible for:

-   loading/training the supported models
-   UQ calibration
-   UQ prediction
-   coverage/breach evaluation
-   UQ-specific artifacts and visualizations
