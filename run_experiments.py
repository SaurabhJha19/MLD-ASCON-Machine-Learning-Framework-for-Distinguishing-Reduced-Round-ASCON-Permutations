import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUN_DIR = ROOT / "run_experiments_result"
DATASETS_RUN_DIR = ROOT / "run_experiments_datasets"

DATASET_SCRIPTS = [
    "datasets.random_dataset",
    "datasets.differential_dataset",
    "datasets.integral_dataset",
    "datasets.cube_dataset",
    "datasets.intermediate_dataset",
]

EXPERIMENT_SCRIPTS = [
    "experiments.accuracy_vs_rounds",
    "experiments.differential_propagation",
    "experiments.intermediate_decay",
    "experiments.feature_importance",
    "experiments.importance_heatmap",
    "experiments.feature_comparison",
    "experiments.model_comparison",
    "experiments.rounds_comparison",
    "experiments.statistical_validation",
    "experiments.statistical_model_benchmark",
    "experiments.cross_round_generalization",
    "experiments.cross_word_generalization",
    "experiments.differential_patterns",
    "experiments.shap_analysis",
    "experiments.diffusion_threshold",
    "experiments.classical_differential",
    "experiments.statistical_classical",
    "experiments.explainability_diffusion",
]


def utc_timestamp():
    return datetime.now(timezone.utc).isoformat()


def snapshot_results(results_dir):
    snapshot = {}

    if not results_dir.exists():
        return snapshot

    for path in results_dir.rglob("*"):
        if not path.is_file():
            continue

        relative = str(path.relative_to(results_dir))

        try:
            stat = path.stat()
            snapshot[relative] = {
                "mtime_ns": stat.st_mtime_ns,
                "size": stat.st_size,
            }
        except OSError:
            pass

    return snapshot


def collect_changed_files(results_dir, before):
    changed = []

    if not results_dir.exists():
        return changed

    for path in results_dir.rglob("*"):
        if not path.is_file():
            continue

        relative = str(path.relative_to(results_dir))

        try:
            stat = path.stat()
            current = {
                "mtime_ns": stat.st_mtime_ns,
                "size": stat.st_size,
            }
        except OSError:
            continue

        if relative not in before or before[relative] != current:
            changed.append(relative)

    return changed


def copy_artifacts(source_results_dir, relative_files, destination_root):
    copied = []

    for relative in relative_files:
        source = source_results_dir / relative

        if not source.exists():
            continue

        target = destination_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

        copied.append(str(target.relative_to(destination_root)))

    return copied


def run_intermediate_dataset_stage(workspace):
    start = time.perf_counter()
    started_at = utc_timestamp()

    generated = []
    stdout_parts = []
    stderr_parts = []

    for target_round in range(1, 5):
        if target_round == 1:
            output_file = "results/intermediate_r4_round1.csv"
        else:
            output_file = f"results/intermediate_round{target_round}.csv"

        command = [
            sys.executable,
            "-c",
            (
                "from datasets.intermediate_dataset import "
                "generate_intermediate_dataset; "
                f"generate_intermediate_dataset("
                f"rounds=4, target_round={target_round}, "
                f"samples=20000, "
                f"output_file={output_file!r})"
            ),
        ]

        env = os.environ.copy()
        root_string = str(ROOT)
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            root_string
            + (os.pathsep + existing_pythonpath
               if existing_pythonpath else "")
        )

        completed = subprocess.run(
            command,
            cwd=workspace,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        stdout_parts.append(
            f"--- target_round={target_round} ---\n"
            + completed.stdout
        )

        if completed.stderr:
            stderr_parts.append(
                f"--- target_round={target_round} ---\n"
                + completed.stderr
            )

        if completed.returncode != 0:
            return {
                "module": "datasets.intermediate_dataset",
                "command": command,
                "working_directory": str(workspace),
                "started_at": started_at,
                "finished_at": utc_timestamp(),
                "duration_seconds": round(
                    time.perf_counter() - start, 3
                ),
                "return_code": completed.returncode,
                "status": "failed",
                "stdout": "\n".join(stdout_parts),
                "stderr": "\n".join(stderr_parts),
                "generated_files": generated,
            }

        generated.append(output_file)

        if target_round == 1:
            source = workspace / "results/intermediate_r4_round1.csv"
            canonical = workspace / "results/intermediate_round1.csv"
            shutil.copy2(source, canonical)
            generated.append("results/intermediate_round1.csv")

    return {
        "module": "datasets.intermediate_dataset",
        "command": [
            sys.executable,
            "-c",
            "generate_intermediate_dataset(rounds=4, target_round=1..4)",
        ],
        "working_directory": str(workspace),
        "started_at": started_at,
        "finished_at": utc_timestamp(),
        "duration_seconds": round(
            time.perf_counter() - start, 3
        ),
        "return_code": 0,
        "status": "success",
        "stdout": "\n".join(stdout_parts),
        "stderr": "\n".join(stderr_parts),
        "generated_files": generated,
    }



def run_differential_dataset_stage(workspace):
    start = time.perf_counter()
    started_at = utc_timestamp()

    generated = []
    stdout_parts = []
    stderr_parts = []

    requests = [
        (2, "results/differential_r2.csv"),
        (3, "results/differential_r3.csv"),
        (4, "results/differential_r4.csv"),
        (5, "results/differential_r5.csv"),
    ]

    for target_round, output_file in requests:
        command = [
            sys.executable,
            "-c",
            (
                "from datasets.differential_dataset import generate_dataset; "
                f"generate_dataset("
                f"rounds={target_round}, "
                f"samples=10000, "
                f"output_file={output_file!r})"
            ),
        ]

        env = os.environ.copy()
        root_string = str(ROOT)
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            root_string
            + (
                os.pathsep + existing_pythonpath
                if existing_pythonpath
                else ""
            )
        )

        completed = subprocess.run(
            command,
            cwd=workspace,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        stdout_parts.append(
            f"--- rounds={target_round} ---\n"
            + completed.stdout
        )

        if completed.stderr:
            stderr_parts.append(
                f"--- rounds={target_round} ---\n"
                + completed.stderr
            )

        if completed.returncode != 0:
            return {
                "module": "datasets.differential_dataset",
                "command": command,
                "working_directory": str(workspace),
                "started_at": started_at,
                "finished_at": utc_timestamp(),
                "duration_seconds": round(
                    time.perf_counter() - start,
                    3,
                ),
                "return_code": completed.returncode,
                "status": "failed",
                "stdout": "\n".join(stdout_parts),
                "stderr": "\n".join(stderr_parts),
                "generated_files": generated,
                "requested_rounds": [2, 3, 4, 5],
            }

        generated.append(output_file)

    return {
        "module": "datasets.differential_dataset",
        "command": [
            sys.executable,
            "-c",
            "generate_dataset(rounds=2..5, samples=10000)",
        ],
        "working_directory": str(workspace),
        "started_at": started_at,
        "finished_at": utc_timestamp(),
        "duration_seconds": round(
            time.perf_counter() - start,
            3,
        ),
        "return_code": 0,
        "status": "success",
        "stdout": "\n".join(stdout_parts),
        "stderr": "\n".join(stderr_parts),
        "generated_files": generated,
        "requested_rounds": [2, 3, 4, 5],
    }

def run_differential_pattern_dataset_stage(workspace):
    start = time.perf_counter()
    started_at = utc_timestamp()

    generated = []
    stdout_parts = []
    stderr_parts = []

    requests = [
        ("P1", 0, 0, "results/diff_p1.csv"),
        ("P2", 0, 15, "results/diff_p2.csv"),
        ("P3", 1, 0, "results/diff_p3.csv"),
        ("P4", 2, 31, "results/diff_p4.csv"),
        ("P5", 4, 63, "results/diff_p5.csv"),
    ]

    for pattern_name, diff_word, diff_bit, output_file in requests:

        command = [
            sys.executable,
            "-c",
            (
                "from datasets.differential_dataset import generate_dataset; "
                f"generate_dataset("
                f"rounds=4, "
                f"samples=10000, "
                f"output_file={output_file!r}, "
                f"diff_word={diff_word}, "
                f"diff_bit={diff_bit})"
            ),
        ]

        env = os.environ.copy()

        root_string = str(ROOT)
        existing_pythonpath = env.get("PYTHONPATH", "")

        env["PYTHONPATH"] = (
            root_string
            + (
                os.pathsep + existing_pythonpath
                if existing_pythonpath
                else ""
            )
        )

        completed = subprocess.run(
            command,
            cwd=workspace,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        stdout_parts.append(
            f"--- {pattern_name} "
            f"(x{diff_word}, b{diff_bit}) ---\n"
            + completed.stdout
        )

        if completed.stderr:
            stderr_parts.append(
                f"--- {pattern_name} "
                f"(x{diff_word}, b{diff_bit}) ---\n"
                + completed.stderr
            )

        if completed.returncode != 0:
            return {
                "module": "datasets.differential_dataset",
                "command": command,
                "working_directory": str(workspace),
                "started_at": started_at,
                "finished_at": utc_timestamp(),
                "duration_seconds": round(
                    time.perf_counter() - start,
                    3,
                ),
                "return_code": completed.returncode,
                "status": "failed",
                "stdout": "\n".join(stdout_parts),
                "stderr": "\n".join(stderr_parts),
                "generated_files": generated,
                "requested_patterns": [
                    "P1 (x0,b0)",
                    "P2 (x0,b15)",
                    "P3 (x1,b0)",
                    "P4 (x2,b31)",
                    "P5 (x4,b63)",
                ],
            }

        generated.append(output_file)

    return {
        "module": "datasets.differential_dataset",
        "command": [
            sys.executable,
            "-c",
            (
                "generate_dataset(rounds=4, samples=10000, "
                "diff_word/diff_bit for P1..P5)"
            ),
        ],
        "working_directory": str(workspace),
        "started_at": started_at,
        "finished_at": utc_timestamp(),
        "duration_seconds": round(
            time.perf_counter() - start,
            3,
        ),
        "return_code": 0,
        "status": "success",
        "stdout": "\n".join(stdout_parts),
        "stderr": "\n".join(stderr_parts),
        "generated_files": generated,
        "requested_patterns": [
            "P1 (x0,b0)",
            "P2 (x0,b15)",
            "P3 (x1,b0)",
            "P4 (x2,b31)",
            "P5 (x4,b63)",
        ],
    }

def run_integral_dataset_stage(workspace):
    start = time.perf_counter()
    started_at = utc_timestamp()

    generated = []
    stdout_parts = []
    stderr_parts = []

    requests = [
        ("default", None, "results/integral_r4.csv"),
        ("x0", 0, "results/integral_x0_r4.csv"),
        ("x1", 1, "results/integral_x1_r4.csv"),
        ("x2", 2, "results/integral_x2_r4.csv"),
        ("x3", 3, "results/integral_x3_r4.csv"),
        ("x4", 4, "results/integral_x4_r4.csv"),
    ]

    for name, active_word, output_file in requests:
        active_word_arg = (
            ""
            if active_word is None
            else f", active_word={active_word}"
        )

        command = [
            sys.executable,
            "-c",
            (
                "from datasets.integral_dataset import generate_dataset; "
                f"generate_dataset("
                f"rounds=4, samples=5000"
                f"{active_word_arg}, "
                f"output_file={output_file!r})"
            ),
        ]

        env = os.environ.copy()
        root_string = str(ROOT)
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            root_string
            + (
                os.pathsep + existing_pythonpath
                if existing_pythonpath
                else ""
            )
        )

        completed = subprocess.run(
            command,
            cwd=workspace,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        stdout_parts.append(
            f"--- {name} ---\n"
            + completed.stdout
        )

        if completed.stderr:
            stderr_parts.append(
                f"--- {name} ---\n"
                + completed.stderr
            )

        if completed.returncode != 0:
            return {
                "module": "datasets.integral_dataset",
                "command": command,
                "working_directory": str(workspace),
                "started_at": started_at,
                "finished_at": utc_timestamp(),
                "duration_seconds": round(
                    time.perf_counter() - start,
                    3,
                ),
                "return_code": completed.returncode,
                "status": "failed",
                "stdout": "\n".join(stdout_parts),
                "stderr": "\n".join(stderr_parts),
                "generated_files": generated,
                "requested_active_words": [
                    "default", "x0", "x1", "x2", "x3", "x4"
                ],
            }

        generated.append(output_file)

    return {
        "module": "datasets.integral_dataset",
        "command": [
            sys.executable,
            "-c",
            "generate_dataset(rounds=4, samples=5000, active_word=None,0..4)",
        ],
        "working_directory": str(workspace),
        "started_at": started_at,
        "finished_at": utc_timestamp(),
        "duration_seconds": round(
            time.perf_counter() - start,
            3,
        ),
        "return_code": 0,
        "status": "success",
        "stdout": "\n".join(stdout_parts),
        "stderr": "\n".join(stderr_parts),
        "generated_files": generated,
        "requested_active_words": [
            "default", "x0", "x1", "x2", "x3", "x4"
        ],
    }


def run_script(module_name, workspace):
    start = time.perf_counter()
    started_at = utc_timestamp()

    command = [
        sys.executable,
        "-m",
        module_name,
    ]

    env = os.environ.copy()
    root_string = str(ROOT)

    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        root_string
        + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    )

    try:
        completed = subprocess.run(
            command,
            cwd=workspace,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        return {
            "module": module_name,
            "command": command,
            "working_directory": str(workspace),
            "started_at": started_at,
            "finished_at": utc_timestamp(),
            "duration_seconds": round(
                time.perf_counter() - start,
                3,
            ),
            "return_code": completed.returncode,
            "status": (
                "success"
                if completed.returncode == 0
                else "failed"
            ),
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }

    except Exception as exc:
        return {
            "module": module_name,
            "command": command,
            "working_directory": str(workspace),
            "started_at": started_at,
            "finished_at": utc_timestamp(),
            "duration_seconds": round(
                time.perf_counter() - start,
                3,
            ),
            "return_code": -1,
            "status": "runner_error",
            "stdout": "",
            "stderr": repr(exc),
        }


def save_logs(run_root, module_name, result):
    log_dir = run_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    safe_name = module_name.replace(".", "_")

    stdout_path = log_dir / f"{safe_name}_stdout.txt"
    stderr_path = log_dir / f"{safe_name}_stderr.txt"

    stdout_path.write_text(
        result.get("stdout", ""),
        encoding="utf-8",
    )

    stderr_path.write_text(
        result.get("stderr", ""),
        encoding="utf-8",
    )

    return {
        "stdout": str(stdout_path.relative_to(run_root)),
        "stderr": str(stderr_path.relative_to(run_root)),
    }


def strip_console_output(result):
    result.pop("stdout", None)
    result.pop("stderr", None)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete MLD-ASCON research replication pipeline."
        )
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "Continue executing remaining scripts if a dataset "
            "or experiment fails."
        ),
    )

    args = parser.parse_args()

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    DATASETS_RUN_DIR.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_root = RUN_DIR / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    dataset_root = DATASETS_RUN_DIR / run_id
    dataset_root.mkdir(parents=True, exist_ok=True)

    workspace = run_root / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    workspace_results = workspace / "results"
    workspace_results.mkdir(parents=True, exist_ok=True)

    total_scripts = len(DATASET_SCRIPTS) + len(EXPERIMENT_SCRIPTS)

    manifest = {
        "project": "MLD-ASCON",
        "run_id": run_id,
        "started_at": utc_timestamp(),
        "python": sys.version,
        "executable": sys.executable,
        "repository_root": str(ROOT),
        "execution_workspace": str(workspace),
        "dataset_output_directory": str(dataset_root),
        "result_output_directory": str(run_root),
        "pipeline": [],
    }

    print("=" * 70)
    print("MLD-ASCON REPLICATION PIPELINE")
    print("=" * 70)
    print(f"Run ID: {run_id}")
    print(f"Python: {sys.version.split()[0]}")
    print()

    pipeline_failed = False

    print("=" * 70)
    print("DATASET GENERATION")
    print("=" * 70)

    for index, module_name in enumerate(
        DATASET_SCRIPTS,
        start=1,
    ):
        print(
            f"[{index}/{total_scripts}] "
            f"Running {module_name} ..."
        )

        before = snapshot_results(workspace_results)

        if module_name == "datasets.differential_dataset":

            result = run_differential_dataset_stage(workspace)

            if result["status"] == "success":

                pattern_result = run_differential_pattern_dataset_stage(
                    workspace
                )

                result["generated_files"].extend(
                    pattern_result.get("generated_files", [])
                )

                result["stdout"] += (
                    "\n\n"
                    + pattern_result.get("stdout", "")
                )

                result["stderr"] += (
                    "\n\n"
                    + pattern_result.get("stderr", "")
                )

                if pattern_result["status"] != "success":
                    result["status"] = "failed"
                    result["return_code"] = pattern_result["return_code"]

        elif module_name == "datasets.integral_dataset":
            result = run_integral_dataset_stage(workspace)

        elif module_name == "datasets.intermediate_dataset":
            result = run_intermediate_dataset_stage(workspace)

        else:
            result = run_script(module_name, workspace)

        changed_files = collect_changed_files(
            workspace_results,
            before,
        )

        generated_datasets = copy_artifacts(
            workspace_results,
            changed_files,
            dataset_root / module_name.replace(".", "_"),
        )

        if module_name == "datasets.differential_dataset":
            result["requested_differential_rounds"] = [2, 3, 4, 5]

        if module_name == "datasets.integral_dataset":
            result["requested_active_words"] = [
                "default", "x0", "x1", "x2", "x3", "x4"
            ]

        if module_name == "datasets.intermediate_dataset":
            result["requested_intermediate_rounds"] = [1, 2, 3, 4]

        logs = save_logs(
            run_root,
            module_name,
            result,
        )

        result["stage"] = "dataset_generation"
        result["generated_datasets"] = generated_datasets
        result["generated_artifacts"] = []
        result["stdout_log"] = logs["stdout"]
        result["stderr_log"] = logs["stderr"]

        strip_console_output(result)
        manifest["pipeline"].append(result)

        if result["status"] == "success":
            print(
                f"    ✓ Success "
                f"({result['duration_seconds']:.2f}s)"
            )
        else:
            print(
                f"    ✗ Failed "
                f"(return code {result['return_code']})"
            )

            pipeline_failed = True

            if not args.continue_on_error:
                break

    if not pipeline_failed or args.continue_on_error:
        print()
        print("=" * 70)
        print("EXPERIMENT EXECUTION")
        print("=" * 70)

        dataset_count = len(DATASET_SCRIPTS)

        for experiment_index, module_name in enumerate(
            EXPERIMENT_SCRIPTS,
            start=1,
        ):
            display_index = dataset_count + experiment_index

            print(
                f"[{display_index}/{total_scripts}] "
                f"Running {module_name} ..."
            )

            before = snapshot_results(workspace_results)

            result = run_script(module_name, workspace)

            changed_files = collect_changed_files(
                workspace_results,
                before,
            )

            artifact_destination = (
                run_root
                / "artifacts"
                / module_name.replace(".", "_")
            )

            generated_artifacts = copy_artifacts(
                workspace_results,
                changed_files,
                artifact_destination,
            )

            logs = save_logs(
                run_root,
                module_name,
                result,
            )

            result["stage"] = "experiment"
            result["generated_datasets"] = []
            result["generated_artifacts"] = generated_artifacts
            result["stdout_log"] = logs["stdout"]
            result["stderr_log"] = logs["stderr"]

            strip_console_output(result)
            manifest["pipeline"].append(result)

            if result["status"] == "success":
                print(
                    f"    ✓ Success "
                    f"({result['duration_seconds']:.2f}s)"
                )
            else:
                print(
                    f"    ✗ Failed "
                    f"(return code {result['return_code']})"
                )

                pipeline_failed = True

                if not args.continue_on_error:
                    break

    finished_at = utc_timestamp()
    manifest["finished_at"] = finished_at

    successful = sum(
        item["status"] == "success"
        for item in manifest["pipeline"]
    )

    failed = sum(
        item["status"] != "success"
        for item in manifest["pipeline"]
    )

    manifest["summary"] = {
        "total_expected": total_scripts,
        "total_executed": len(manifest["pipeline"]),
        "successful": successful,
        "failed": failed,
        "pipeline_status": (
            "success"
            if failed == 0
            and len(manifest["pipeline"]) == total_scripts
            else "failed"
        ),
    }

    manifest["outputs"] = {
        "datasets": str(dataset_root.relative_to(ROOT)),
        "results": str(run_root.relative_to(ROOT)),
        "manifest": str(
            (run_root / "run_manifest.json").relative_to(ROOT)
        ),
    }

    manifest_path = run_root / "run_manifest.json"

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    latest_path = RUN_DIR / "latest_run.json"

    latest_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "manifest": str(
                    manifest_path.relative_to(RUN_DIR)
                ),
                "timestamp": finished_at,
                "status": manifest["summary"]["pipeline_status"],
                "datasets": str(dataset_root.relative_to(ROOT)),
                "results": str(run_root.relative_to(ROOT)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if manifest["summary"]["pipeline_status"] == "success":
        shutil.rmtree(workspace, ignore_errors=True)

    print()
    print("=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    print(f"Expected  : {manifest['summary']['total_expected']}")
    print(f"Executed  : {manifest['summary']['total_executed']}")
    print(f"Successful: {manifest['summary']['successful']}")
    print(f"Failed    : {manifest['summary']['failed']}")
    print(f"\nDatasets : {dataset_root}")
    print(f"Results  : {run_root}")
    print(f"Manifest : {manifest_path}")
    print("=" * 70)

    if manifest["summary"]["pipeline_status"] != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()