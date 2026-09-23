# -*- coding: utf-8 -*-
"""
@author: pmarantis

trustworthy-uq - Functional End-to-End Workflow Test

This script demonstrates the main packaged workflow using existing
dataset and model artifacts.

Main workflow:
    Dataset
       ↓
    run_experiment()
       ↓
    Calibration
       ↓
    Test-set evaluation
       ↓
    JSON / CSV / plots
       ↓
    Optional direct single-sample prediction
       ↓
    Optional single-sample plot

No direct evaluate() call is needed here because run_experiment()
already performs the complete test-set evaluation.

Change the configuration variables below to test different workflows.
"""

from pathlib import Path
import json


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

# Dataset
DATASET_PATH = (
    BASE_DIR
    / "data"
    / "nomadic_dataset"
    / "ULA_lab_LoS"
)

# Existing ADN backbone
ADN_CHECKPOINT = (
    BASE_DIR
    / "mlartifacts"
    / "851097306110734214"
    / "c5ce26cc459c457fb37c42b721f203d6"
    / "artifacts"
    / "checkpoints"
    / "best_model.ckpt"
)

# Existing CQR head
CQR_CHECKPOINT = (
    BASE_DIR
    / "483259487935076753"
    / "fa4d15510f974de586300cc61026b010"
    / "checkpoints"
    / "best_cqr_head.ckpt"
)

# Existing aSCP checkpoint, if available.
# Set to None if aSCP should be trained from scratch.
ASCP_CHECKPOINT = None


# -------------------------------------------------------------------------
# Workflow selection
# -------------------------------------------------------------------------

METHOD = "CQR"
# Supported:
#     "CQR"
#     "aSCP"

SLA_LEVELS = [0.90, 0.95, 0.99]

EVAL_MODE = "pooled"

SCENARIO_IDS = range(0, 6)
SCENARIO_ID = 0


# -------------------------------------------------------------------------
# Model workflow
# -------------------------------------------------------------------------

# ADN
TRAIN_NEW_ADN = False

# CQR
TRAIN_NEW_CQR = False

# aSCP
TRAIN_NEW_ASCP = False


# -------------------------------------------------------------------------
# Single-sample functionality
# -------------------------------------------------------------------------

ENABLE_SINGLE_SAMPLE = True

SINGLE_SAMPLE_INDEX = 0
SINGLE_SAMPLE_SLA = 0.95

# Test the separate predict() API after run_experiment()
RUN_DIRECT_PREDICT = True

# Test the separate plot_sample() API
RUN_SINGLE_PLOT = True


# -------------------------------------------------------------------------
# Output
# -------------------------------------------------------------------------

OUTPUT_DIR = BASE_DIR / "package_functional_test"


# =============================================================================
# SMALL HELPERS
# =============================================================================

def check_exists(path, name):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"{name} was not found:\n{path}"
        )

    print(f"[OK] {name}: {path}")


def load_json(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Expected JSON artifact was not created:\n{path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 72)
    print("trustworthy-uq - FUNCTIONAL E2E TEST")
    print("=" * 72)

    # -------------------------------------------------------------------------
    # 1. Import package
    # -------------------------------------------------------------------------

    import trustworthy_uq
    from trustworthy_uq.localization.pipeline import LocalizationUQ

    print("\n[1] Package")
    print(f"[OK] trustworthy_uq: {trustworthy_uq.__file__}")

    # -------------------------------------------------------------------------
    # 2. Check required artifacts
    # -------------------------------------------------------------------------

    print("\n[2] Input artifacts")

    check_exists(DATASET_PATH, "Dataset")
    check_exists(ADN_CHECKPOINT, "ADN checkpoint")

    method_upper = METHOD.upper()

    if method_upper == "CQR":

        if not TRAIN_NEW_CQR:
            check_exists(CQR_CHECKPOINT, "CQR checkpoint")

    elif method_upper in ("ASCP", "ADAPTIVESCP"):

        if not TRAIN_NEW_ASCP:

            if ASCP_CHECKPOINT is None:
                raise ValueError(
                    "For aSCP, either set TRAIN_NEW_ASCP=True "
                    "or provide ASCP_CHECKPOINT."
                )

            check_exists(ASCP_CHECKPOINT, "aSCP checkpoint")

    else:
        raise ValueError(
            f"Unsupported METHOD='{METHOD}'. "
            "Use 'CQR' or 'aSCP'."
        )

    # -------------------------------------------------------------------------
    # 3. Create LocalizationUQ
    # -------------------------------------------------------------------------

    print("\n[3] Creating LocalizationUQ")

    uq = LocalizationUQ(
        sla_levels=SLA_LEVELS,
        seed=42,
        batch_size=32,
        num_workers=0,
    )

    print("[OK] LocalizationUQ created")

    # -------------------------------------------------------------------------
    # 4. Run complete experiment
    # -------------------------------------------------------------------------

    print("\n[4] Running complete experiment")

    experiment_output = OUTPUT_DIR / "run_experiment"

    experiment_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"  Method:      {METHOD}")
    print(f"  SLA levels:  {SLA_LEVELS}")
    print(f"  Eval mode:   {EVAL_MODE}")
    print(f"  Output:      {experiment_output}")

    experiment_result = uq.run_experiment(
        data_dir=str(DATASET_PATH),

        method=METHOD,
        eval_mode=EVAL_MODE,

        scenario_id=SCENARIO_ID,
        scenario_ids=SCENARIO_IDS,

        adn_checkpoint=(
            str(ADN_CHECKPOINT)
            if not TRAIN_NEW_ADN
            else None
        ),

        cqr_checkpoint=(
            str(CQR_CHECKPOINT)
            if (
                method_upper == "CQR"
                and not TRAIN_NEW_CQR
            )
            else None
        ),

        ascp_checkpoint=(
            str(ASCP_CHECKPOINT)
            if (
                method_upper in ("ASCP", "ADAPTIVESCP")
                and not TRAIN_NEW_ASCP
                and ASCP_CHECKPOINT is not None
            )
            else None
        ),

        train_new_adn=TRAIN_NEW_ADN,
        train_new_cqr=TRAIN_NEW_CQR,
        train_new_ascp=TRAIN_NEW_ASCP,

        sla_levels=SLA_LEVELS,

        output_dir=str(experiment_output),

        enable_single_sample=ENABLE_SINGLE_SAMPLE,
        single_sample_index=SINGLE_SAMPLE_INDEX,
        single_sample_sla=SINGLE_SAMPLE_SLA,
    )

    print("[OK] run_experiment completed")

    # -------------------------------------------------------------------------
    # 5. Locate generated artifacts
    # -------------------------------------------------------------------------

    actual_output = (
        experiment_output
        / (
            "pooled_scenarios"
            if EVAL_MODE.lower() == "pooled"
            else f"single_scenario_{SCENARIO_ID}"
        )
        / METHOD.upper()
    )

    check_exists(
        actual_output,
        "Experiment output directory",
    )

    print(f"\n[5] Generated artifacts: {actual_output}")

    generated_files = sorted(
        p.relative_to(actual_output)
        for p in actual_output.rglob("*")
        if p.is_file()
    )

    for file in generated_files:
        print(f"  - {file}")

    # -------------------------------------------------------------------------
    # 6. Verify main JSON artifacts
    # -------------------------------------------------------------------------

    print("\n[6] Verifying JSON artifacts")

    calibration_file = (
        actual_output
        / (
            "cqr_calibration.json"
            if method_upper == "CQR"
            else "ascp_calibration.json"
        )
    )

    test_results_file = (
        actual_output
        / "uq_test_results.json"
    )

    single_sample_file = (
        actual_output
        / "uq_single_sample.json"
    )

    calibration = load_json(calibration_file)
    test_results = load_json(test_results_file)

    print(
        f"[OK] Calibration JSON: "
        f"{calibration_file.name}"
    )

    print(
        f"[OK] Test-results JSON: "
        f"{test_results_file.name}"
    )

    if ENABLE_SINGLE_SAMPLE:

        single_sample = load_json(single_sample_file)

        print(
            f"[OK] Single-sample JSON: "
            f"{single_sample_file.name}"
        )

    # -------------------------------------------------------------------------
    # 7. Print compact test-set summary
    # -------------------------------------------------------------------------

    print("\n[7] Test-set UQ results")

    for sla, result in test_results.get(
        "sla_results",
        {}
    ).items():

        coverage = result.get("true_coverage")
        breach = result.get("breach_rate")
        radius = result.get("mean_final_radius_mm")

        print(
            f"  SLA={sla}: "
            f"coverage={coverage:.4f}, "
            f"breach={breach:.4f}, "
            f"mean radius={radius:.2f} mm"
        )

    # -------------------------------------------------------------------------
    # 8. Direct single-sample prediction
    # -------------------------------------------------------------------------

    if RUN_DIRECT_PREDICT:

        print("\n[8] Direct single-sample prediction")

        # Reconstruct the same dataset used by the experiment.
        dataset = uq.build_pooled_dataset(
            data_dir=str(DATASET_PATH),
            scenario_ids=SCENARIO_IDS,
            num_users=4,
            num_samples=240,
        )

        _, _, test_dataset = uq.split_dataset(dataset)

        X_sample, y_sample = test_dataset[
            SINGLE_SAMPLE_INDEX
        ]

        prediction = uq.predict(
            samples=X_sample,
            y_true=y_sample,
            sla_levels=SLA_LEVELS,
            method=METHOD,
        )

        print("[OK] Prediction completed")

        print(
            "  Point prediction:",
            prediction["point_prediction"]
        )

        print(
            "  Radius (mm):",
            prediction["radius_mm"]
        )

        print(
            "  Covered:",
            prediction["covered"]
        )

    # -------------------------------------------------------------------------
    # 9. Single-sample plotting
    # -------------------------------------------------------------------------

    if RUN_SINGLE_PLOT:

        print("\n[9] Single-sample plot")

        # X_sample/y_sample are already available if direct prediction
        # was executed. Otherwise construct the test sample here.
        if not RUN_DIRECT_PREDICT:

            dataset = uq.build_pooled_dataset(
                data_dir=str(DATASET_PATH),
                scenario_ids=SCENARIO_IDS,
                num_users=4,
                num_samples=240,
            )

            _, _, test_dataset = uq.split_dataset(dataset)

            X_sample, y_sample = test_dataset[
                SINGLE_SAMPLE_INDEX
            ]

        plot_path = (
            actual_output
            / "additional_single_sample_test"
        )

        try:

            uq.plot_sample(
                samples=X_sample,
                y_true=y_sample,
                sample_index=0,
                sla_level=SINGLE_SAMPLE_SLA,
                method=METHOD,
                output_path=str(plot_path),
            )

            print("[OK] Single-sample plot created")

        except Exception as exc:

            # Plotting is supplementary; don't classify a Plotly/Kaleido
            # problem as a failure of the UQ workflow.
            print(
                f"[WARNING] Single-sample plotting failed: {exc}"
            )

    # -------------------------------------------------------------------------
    # 10. Final summary
    # -------------------------------------------------------------------------

    print("\n" + "=" * 72)
    print("FUNCTIONAL E2E TEST COMPLETED")
    print("=" * 72)

    print(f"Method:       {METHOD}")
    print(f"SLA levels:   {SLA_LEVELS}")
    print(f"Output:       {actual_output}")

    print("\nMain MLOps artifacts:")

    print(
        f"  Test results:      {test_results_file}"
    )

    print(
        f"  Calibration:       {calibration_file}"
    )

    if ENABLE_SINGLE_SAMPLE:

        print(
            f"  Single sample:     {single_sample_file}"
        )

    print("\n[OK] Workflow completed successfully.")


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    main()