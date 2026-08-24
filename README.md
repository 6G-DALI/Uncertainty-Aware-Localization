#### \### Uncertainty-Aware CSI Localization Experiments



This repository contains the main experiment scripts used in the paper "SLA-Aware Conformal Uncertainty for Trustworthy CSI Localization in 6G". The script `Trustworthy\_Localization\_Experiments\_full\_modular.py`

supports execution of the individual experiment workflows (Adaptive SCP, CQR, pooled and per-scenario experiments) in a single modular file.



###### \#### Project scope and preparation



The paper experiments focus on three main settings:

\- Pooled dynamic-scenario evaluation of Adaptive SCP with a learned scale model.

\- Pooled comparison of SCP and CQR under shared splits.

\- Per-scenario CQR evaluation for distinct dynamic scenarios.



The experiments use an AttentionDenseNet (ADN) backbone and evaluate conformal uncertainty methods on dynamic localization scenarios, with logging and artifact tracking through MLflow.

Attention Dense Net and the data preprocessing/augmentation methods are derived from: L. Schuhmacher, H. Sallouha, I. Gryech, and S. Pollin, "Data Augmentation and Attention for Massive MIMO-based Indoor Localization in Changing Environments," IEEE International Conference on Communications (ICC), 2026.



The ULA subset of the nomadic dataset from https://ieee-dataport.org/open-access/ultra-dense-indoor-mamimo-csi-dataset is used as data. Download and place the `nomadic\_dataset` directory under `data/` (resulting in `data/nomadic\_dataset/ULA\_lab\_LoS`).

Use `pip install -r requirements.txt` to prepare a virtual environment for running the repository.

Set up MLflow tracking URI (in a separate shell):

```bash

mlflow server --host 127.0.0.1 --port 8080

```

The unified script logs runs to the experiment name `ADN-Conformal-Unified` and uses run names derived from the selected method and workflow. 



###### \#### Unified script: Trustworthy\_Localization\_Experiments\_full\_modular.py



The `Trustworthy\_Localization\_Experiments\_full\_modular.py` script allows running all the conformal uncertainty experiments used in the paper.

It supports:

\- Methods: Adaptive Split Conformal Prediction (Adaptive SCP with a learned scale model) and one-sided multi-SLA CQR on top of a frozen ADN backbone.

\- Evaluation modes: pooled dynamic scenarios (0–5) or a single dynamic scenario.

\- CQR head handling: train from scratch or reuse an existing CQR head checkpoint.

\- ADN backbone handling: reuse a pretrained backbone checkpoint or train a new backbone from static data with selectable augmentation.

\- Metrics: overall SLA coverage and uncertainty metrics on the full test set, plus optional single-sample evaluation with a per-sample prediction disk plot.



###### \##### Core configuration variables



At the top of the script, several user-configurable variables define the workflow:



\- `METHOD`: chooses the conformal method.

&#x20; - `"ASCP"`: Adaptive SCP with learned scale model.

&#x20; - `"CQR"`: one-sided multi-SLA CQR head.



\- `EVAL\_MODE`: chooses the evaluation data.

&#x20; - `"pooled"`: pooled dynamic scenarios `\[0, 1, 2, 3, 4, 5]`.

&#x20; - `"single"`: one dynamic scenario; selected via `SINGLE\_SCENARIO\_ID`.



\- `SINGLE\_SCENARIO\_ID`: integer in `\[0, 5]`, used when `EVAL\_MODE == "single"`.



\- `USE\_EXISTING\_BACKBONE\_CHECKPOINT` and `BACKBONE\_CHECKPOINT\_PATH`:

&#x20; - When `True`, the script loads a pretrained ADN backbone from `BACKBONE\_CHECKPOINT\_PATH`.

&#x20; - When `False`, the script can train a new ADN backbone from static data, controlled by `TRAIN\_ADN\_IF\_NEEDED` and `ADN\_AUGMENT\_METHOD`.



\- `TRAIN\_ADN\_IF\_NEEDED` (default `False`):

&#x20; - If `USE\_EXISTING\_BACKBONE\_CHECKPOINT == False` and `TRAIN\_ADN\_IF\_NEEDED == True`, the script trains a new ADN backbone on the static scenario with the selected augmentation method.



\- `ADN\_AUGMENT\_METHOD` (default `"random\_attenuation"`):

&#x20; - `"random\_attenuation"`: random attenuation augmentation (best-performing setting in the paper).

&#x20; - `"vanilla"`: vanilla antenna-blocking augmentation.

&#x20; - `"no\_augmentation"`: no augmentation.



\- `USE\_EXISTING\_CQR\_HEAD` and `CQR\_HEAD\_CHECKPOINT\_PATH`:

&#x20; - When `True`, the script loads an existing CQR head checkpoint for the selected backbone.

&#x20; - When `False`, the script trains a new CQR head on the chosen dataset split.\[2]



\- `SLA\_LEVELS`: list of target assurance levels (default `\[0.90, 0.95, 0.99]`).



\- `HEAD\_TRAIN\_FRACTION`, `CALIBRATION\_FRACTION`, `TEST\_FRACTION`:

&#x20; - Fractions of the pooled or scenario-specific dataset used for head/scale training, conformal calibration, and testing, respectively (default `0.25`, `0.45`, `0.30`).



\- `ENABLE\_SINGLE\_SAMPLE\_EVAL`, `SINGLE\_SAMPLE\_INDEX`, `SINGLE\_SAMPLE\_SLA`:

&#x20; - Control the optional per-sample evaluation workflow, where a single test sample is visualized with a prediction disk and true location.



Other variables such as `SEED`, `DATA\_DIR`, `TRACKING\_URI`, `EXPERIMENT\_NAME`, and `RUN\_NAME` configure reproducibility, data paths, and MLflow logging.



###### \##### Supported workflows



The main workflows correspond to the paper’s experiments and are now unified under `Trustworthy\_Localization\_Experiments\_full\_modular.py`.



1\. \*\*Adaptive SCP on pooled dynamic scenarios\*\*

&#x20;  - Set:

&#x20;    - `METHOD = "ASCP"`

&#x20;    - `EVAL\_MODE = "pooled"`

&#x20;    - Choose `USE\_EXISTING\_BACKBONE\_CHECKPOINT` and `BACKBONE\_CHECKPOINT\_PATH` (or enable `TRAIN\_ADN\_IF\_NEEDED`).

&#x20;  - Run:

&#x20;  ```bash

&#x20;  python Trustworthy\_Localization\_Experiments\_full\_modular.py

&#x20;  ```

&#x20;  The script:

&#x20;  - Loads or trains an ADN backbone.

&#x20;  - Builds a pooled dataset over scenarios 0–5.

&#x20;  - Splits into head-train, calibration, and test subsets.

&#x20;  - Trains a scale model on frozen backbone features.

&#x20;  - Calibrates adaptive radii for `SLA\_LEVELS`.

&#x20;  - Logs metrics, prediction CSVs, and coverage/radius figures to `output/unified\_conformal\_experiments/pooled\_scenarios/ASCP` and MLflow.



2\. \*\*CQR on pooled dynamic scenarios (SCP vs CQR comparison)\*\*

&#x20;  - Set:

&#x20;    - `METHOD = "CQR"`

&#x20;    - `EVAL\_MODE = "pooled"`

&#x20;    - Configure backbone and CQR head handling (e.g., load existing CQR head or train from scratch).

&#x20;  - Run:

&#x20;  ```bash

&#x20;  python Trustworthy\_Localization\_Experiments\_full\_modular.py

&#x20;  ```

&#x20;  The script:

&#x20;  - Loads or trains an ADN backbone.

&#x20;  - Loads or trains a multi-SLA one-sided CQR head.

&#x20;  - Splits the pooled dynamic dataset into head-train, calibration, and test subsets.

&#x20;  - Applies conformal calibration to obtain SLA-specific radii.

&#x20;  - Computes empirical coverage and mean radii per SLA.

&#x20;  - Logs metrics, prediction CSVs, and figures under `output/unified\_conformal\_experiments/pooled\_scenarios/CQR` and to MLflow.



3\. \*\*Per-scenario CQR experiments\*\*

&#x20;  - Set:

&#x20;    - `METHOD = "CQR"`

&#x20;    - `EVAL\_MODE = "single"`

&#x20;    - `SINGLE\_SCENARIO\_ID` to the desired dynamic scenario ID (0–5).

&#x20;  - Run:

&#x20;  ```bash

&#x20;  python Trustworthy\_Localization\_Experiments\_full\_modular.py

&#x20;  ```

&#x20;  The script:

&#x20;  - Builds a dataset for the chosen scenario.

&#x20;  - Splits into head-train, calibration, and test within that scenario.

&#x20;  - Trains or loads a per-scenario CQR head.

&#x20;  - Calibrates SLA-specific radii and computes scenario-level coverage and radius metrics.

&#x20;  - Logs metrics and predictions under `output/unified\_conformal\_experiments/single\_scenario\_<id>/CQR` and MLflow.\[2]



4\. \*\*Training a new ADN backbone with selected augmentation\*\*



&#x20;  - Set:

&#x20;    - `USE\_EXISTING\_BACKBONE\_CHECKPOINT = False`

&#x20;    - `TRAIN\_ADN\_IF\_NEEDED = True`

&#x20;    - `ADN\_AUGMENT\_METHOD` to one of `"random\_attenuation"`, `"vanilla"`, or `"no\_augmentation"`.

&#x20;  - Run the script with any of the above workflows.



&#x20;  The script:

&#x20;  - Trains AttentionDenseNet on the static scenario in `mode="train\_static"` using the selected augmentation method.

&#x20;  - Uses early stopping and MLflow logging (run name `unified\_conformal\_experiments-ADN-train-<augment\_method>`).

&#x20;  - Reloads the best ADN checkpoint and uses it as the backbone for the conformal experiments.\[2]



###### \##### Single-sample uncertainty visualization



The unified script optionally supports a per-sample evaluation workflow that visualizes a single test sample’s prediction disk and true location:\[2]



\- Controlled by:

&#x20; - `ENABLE\_SINGLE\_SAMPLE\_EVAL` (default `True`).

&#x20; - `SINGLE\_SAMPLE\_INDEX`: index in the test split.

&#x20; - `SINGLE\_SAMPLE\_SLA`: SLA level at which to draw the prediction disk.



\- For Adaptive SCP:

&#x20; - The script reconstructs the final radii per SLA from the stored predictions, then calls `plot\_single\_sample\_uncertainty` to generate a 2D plot.

\- For CQR:

&#x20; - The script combines base CQR radii and conformal corrections to obtain final radii per SLA, then plots the selected sample’s disk.



Plots are stored under `output/unified\_conformal\_experiments/.../figures\_single\_samples` and logged as MLflow artifacts in nested runs named `single\_sample\_ASCP\_\*` or `single\_sample\_CQR\_\*`.



###### \#### Set backbone checkpoints 



For reproducibility of the original paper experiments, you can use explicit ADN backbone checkpoints or ADN CQR head checkpoints without training:

\# random attenuation augmentation checkpoint model (best)

\# BACKBONE\_CHECKPOINT\_PATH = r".\\mlartifacts\\851097306110734214\\c5ce26cc459c457fb37c42b721f203d6\\artifacts\\checkpoints\\best\_model.ckpt"



\# vanilla augmentation checkpoint model

\# BACKBONE\_CHECKPOINT\_PATH = r".\\mlartifacts\\851097306110734214\\8a31fb1083504eb580e14437b0002d5c\\artifacts\\checkpoints\\best\_model.ckpt"



\# no augmentation checkpoint model

BACKBONE\_CHECKPOINT\_PATH = r".\\mlartifacts\\851097306110734214\\4b7c1e2582a04239ab3fbfe0957605c3\\artifacts\\\\checkpoints\\best\_model.ckpt"



Example for CQR trained head backbone:
USE\_EXISTING\_CQR\_HEAD: bool = True 

CQR\_HEAD\_CHECKPOINT\_PATH: Optional\[str] = r'./483259487935076753/fa4d15510f974de586300cc61026b010/checkpoints/best\_cqr\_head.ckpt'



For the best-performing paper setting, use the random attenuation checkpoint. The no-augmentation and vanilla checkpoints can be selected to reproduce the corresponding ablation results.



###### \#### Recommended execution order



To reproduce the paper experiments efficiently using the unified script:



1\. Start the MLflow server.

2\. Set `DATA\_DIR`, `TRACKING\_URI`, and the desired ADN backbone configuration.

3\. Run `Trustworthy\_Localization\_Experiments\_full\_modular.py` with `METHOD = "ASCP"`, `EVAL\_MODE = "pooled"` for pooled Adaptive SCP.

4\. Run `Trustworthy\_Localization\_Experiments\_full\_modular.py` with `METHOD = "CQR"`, `EVAL\_MODE = "pooled"` for pooled CQR.

5\. Run per-scenario CQR experiments with `METHOD = "CQR"`, `EVAL\_MODE = "single"`, and varying `SINGLE\_SCENARIO\_ID` for each dynamic scenario of interest.



\#### Scenario-specific notes



The scenario-level datasets use dynamic scenario IDs and sample construction based on: 4 users, 240 samples per user, Scenario-specific sample IDs derived from `scenario\_id \* 10000 + user \* 1000 + sample\_index`. The pooled workflows combine all six dynamic scenarios: SCENARIO\_IDS = \[0, 1, 2, 3, 4, 5]. The pooled experiments in the paper are designed around fixed shared splits and frozen or pretrained ADN backbones, so changing checkpoints or split fractions will change the reported metrics. For exact reproduction, keep the configured split ratios, scenario sets, SLA levels, and checkpoint selection aligned with the values used in the final experiments.



###### \#### Modular use with APIs



The loaded dataset, ADN backbone model, and CQR trained head model can all be trained from scratch or loaded via APIs as artifacts, instead of the current manual configuration.

Similarly, the output metrics and artifacts are all saved both locally and on the MLflow server, and can be retrieved via APIs as needed. Further enhancing a demo-style run, the single-test sample workflow can be used to provide uncertainty artifacts (plots and metrics) for the localization of streaming input test samples for a selected SLA target (e.g., 90, 95, or 99 % coverage). 









