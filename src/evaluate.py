import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from omegaconf import OmegaConf
from scipy.stats import ttest_ind

matplotlib.use("Agg")

PRIMARY_METRIC = "accuracy"
METRIC_DIRECTION = {
    "accuracy": "higher",
    "mean_abs_error": "lower",
    "parse_success_rate": "higher",
    "state_valid_rate": "higher",
    "repair_rate": "lower",
    "avg_model_calls": "lower",
    "avg_output_tokens": "lower",
    "avg_repairs_per_problem": "lower",
}


def parse_args() -> argparse.Namespace:
    raw_args = sys.argv[1:]
    normalized: List[str] = []
    for arg in raw_args:
        if arg.startswith("--") and "=" in arg:
            key, value = arg[2:].split("=", 1)
            normalized.extend([f"--{key}", value])
        elif "=" in arg and not arg.startswith("--"):
            key, value = arg.split("=", 1)
            normalized.extend([f"--{key}", value])
        else:
            normalized.append(arg)
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--run_ids", type=str, required=True)
    return parser.parse_args(normalized)


def to_serializable(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, default=to_serializable)


def metric_is_lower_better(metric_name: str) -> bool:
    direction = METRIC_DIRECTION.get(metric_name, None)
    if direction is not None:
        return direction == "lower"
    lower_keywords = ["loss", "error", "perplexity", "mae"]
    name = metric_name.lower()
    return any(k in name for k in lower_keywords)


def plot_line_metric(df: pd.DataFrame, metric: str, out_path: Path, title: str) -> bool:
    if metric not in df.columns:
        return False
    series = df[metric]
    if series.dropna().empty:
        return False
    x = df["_step"] if "_step" in df.columns else pd.Series(range(len(df)))
    mask = series.notna()
    x_vals = x[mask]
    y_vals = series[mask]
    plt.figure(figsize=(7, 4))
    plt.plot(x_vals, y_vals, label=metric)
    plt.title(title)
    plt.xlabel("step")
    plt.ylabel(metric)
    last_val = y_vals.iloc[-1]
    plt.annotate(f"{last_val:.4f}", (x_vals.iloc[-1], last_val))
    if len(y_vals) > 1:
        best_idx = y_vals.idxmin() if metric_is_lower_better(metric) else y_vals.idxmax()
        best_val = y_vals.loc[best_idx]
        if best_idx != y_vals.index[-1]:
            plt.scatter([x.loc[best_idx]], [best_val], color="red", zorder=3)
            plt.annotate(f"best {best_val:.4f}", (x.loc[best_idx], best_val))
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True


def plot_error_hist(df: pd.DataFrame, out_path: Path, title: str) -> bool:
    if "example_abs_error" not in df.columns:
        return False
    errors = df["example_abs_error"].dropna().values
    if len(errors) == 0:
        return False
    plt.figure(figsize=(7, 4))
    plt.hist(np.log10(errors + 1e-9), bins=30, color="#4c72b0")
    plt.title(title)
    plt.xlabel("log10(|error|+1e-9)")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True


def plot_boxplot(df: pd.DataFrame, column: str, out_path: Path, title: str) -> bool:
    if column not in df.columns:
        return False
    values = df[column].dropna().values
    if len(values) == 0:
        return False
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=values, color="#55a868")
    plt.title(title)
    plt.xlabel(column)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True


def plot_histogram(df: pd.DataFrame, column: str, out_path: Path, title: str) -> bool:
    if column not in df.columns:
        return False
    values = df[column].dropna().values
    if len(values) == 0:
        return False
    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=10, color="#c44e52")
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True


def plot_parse_success_bar(df: pd.DataFrame, out_path: Path, title: str) -> bool:
    if "example_parse_success" not in df.columns:
        return False
    vals = df["example_parse_success"].dropna().values
    if len(vals) == 0:
        return False
    success = int(np.sum(vals))
    failure = int(len(vals) - success)
    plt.figure(figsize=(5, 4))
    plt.bar(["parse_success", "parse_failure"], [success, failure], color=["#4c72b0", "#c44e52"])
    for i, v in enumerate([success, failure]):
        plt.text(i, v, str(v), ha="center", va="bottom")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True


def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, out_path: Path, title: str) -> bool:
    if x_col not in df.columns or y_col not in df.columns:
        return False
    data = df[[x_col, y_col]].dropna()
    if data.empty:
        return False
    plt.figure(figsize=(6, 4))
    plt.scatter(data[x_col], data[y_col], alpha=0.6)
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True


def plot_complexity_accuracy(df: pd.DataFrame, out_path: Path, title: str) -> bool:
    required = {"example_complexity", "example_correct"}
    if not required.issubset(df.columns):
        return False
    data = df[list(required)].dropna()
    if data.empty:
        return False
    bins = {
        "1-2": data[data["example_complexity"] <= 2],
        "3-4": data[(data["example_complexity"] > 2) & (data["example_complexity"] <= 4)],
        "5+": data[data["example_complexity"] > 4],
    }
    accuracies = {}
    for name, subset in bins.items():
        accuracies[name] = subset["example_correct"].mean() if not subset.empty else float("nan")
    labels = list(accuracies.keys())
    values = [accuracies[k] for k in labels]
    plt.figure(figsize=(6, 4))
    sns.barplot(x=labels, y=values, color="#4c72b0")
    plt.title(title)
    plt.ylabel("accuracy")
    for i, v in enumerate(values):
        if not math.isnan(v):
            plt.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True


def plot_confusion_matrix(df: pd.DataFrame, out_path: Path, title: str) -> bool:
    required = {"example_parse_success", "example_correct"}
    if not required.issubset(df.columns):
        return False
    data = df[list(required)].dropna()
    if data.empty:
        return False
    parse_vals = data["example_parse_success"].astype(int).values
    correct_vals = data["example_correct"].astype(int).values
    matrix = np.zeros((2, 2), dtype=int)
    for p, c in zip(parse_vals, correct_vals):
        if p in (0, 1) and c in (0, 1):
            matrix[p, c] += 1
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["incorrect", "correct"],
        yticklabels=["parse_fail", "parse_success"],
    )
    plt.title(title)
    plt.xlabel("correctness")
    plt.ylabel("parse success")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True


def aggregate_metrics(summaries: Dict[str, Dict[str, Any]], primary_metric: str) -> Dict[str, Any]:
    metric_names = set()
    for summary in summaries.values():
        for key, value in summary.items():
            if isinstance(value, (int, float, np.floating, np.integer)):
                if not (isinstance(value, float) and math.isnan(value)):
                    metric_names.add(key)
    metrics = {name: {} for name in metric_names}
    for run_id, summary in summaries.items():
        for name in metric_names:
            if name in summary:
                metrics[name][run_id] = summary[name]

    proposed = {k: v for k, v in summaries.items() if "proposed" in k}
    baseline = {k: v for k, v in summaries.items() if "comparative" in k or "baseline" in k}

    lower_better = metric_is_lower_better(primary_metric)

    def best_run(group: Dict[str, Dict[str, Any]]) -> Any:
        if not group:
            return None
        if lower_better:
            return min(group.items(), key=lambda x: x[1].get(primary_metric, float("inf")))
        return max(group.items(), key=lambda x: x[1].get(primary_metric, float("-inf")))

    best_proposed = best_run(proposed)
    best_baseline = best_run(baseline)

    gap = None
    if best_proposed and best_baseline:
        p_val = best_proposed[1].get(primary_metric, 0.0)
        b_val = best_baseline[1].get(primary_metric, 0.0)
        if b_val not in (0.0, None):
            gap = (p_val - b_val) / b_val * 100
            if lower_better:
                gap = -gap

    return {
        "primary_metric": primary_metric,
        "metrics": metrics,
        "best_proposed": {
            "run_id": best_proposed[0],
            "value": best_proposed[1].get(primary_metric),
        }
        if best_proposed
        else None,
        "best_baseline": {
            "run_id": best_baseline[0],
            "value": best_baseline[1].get(primary_metric),
        }
        if best_baseline
        else None,
        "gap": gap,
    }


def plot_comparison_bar(values: Dict[str, float], out_path: Path, title: str, ylabel: str) -> bool:
    if not values:
        return False
    labels = list(values.keys())
    vals = [values[k] for k in labels]
    plt.figure(figsize=(8, 4))
    sns.barplot(x=labels, y=vals)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    for i, v in enumerate(vals):
        if not math.isnan(v):
            plt.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True


def plot_comparison_boxplot(values: Dict[str, np.ndarray], out_path: Path, title: str, ylabel: str) -> bool:
    if not values:
        return False
    records = []
    for run_id, vals in values.items():
        for val in vals:
            records.append({"run_id": run_id, "value": val})
    df = pd.DataFrame(records)
    if df.empty:
        return False
    plt.figure(figsize=(9, 4))
    sns.boxplot(data=df, x="run_id", y="value")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True


def render_metrics_table(df: pd.DataFrame, out_path: Path, title: str) -> bool:
    if df.empty:
        return False
    plt.figure(figsize=(10, max(2, 0.5 * len(df))))
    plt.axis("off")
    table = plt.table(
        cellText=df.values,
        colLabels=df.columns,
        rowLabels=df.index,
        loc="center",
        cellLoc="center",
    )
    table.scale(1, 1.5)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True


def check_run_is_full(config: Dict[str, Any], run_id: str) -> None:
    wandb_cfg = config.get("wandb") if isinstance(config.get("wandb"), dict) else None
    wandb_mode = config.get("wandb.mode") if "wandb.mode" in config else None
    mode = config.get("mode")
    disabled = False
    if wandb_mode is not None and str(wandb_mode).lower() == "disabled":
        disabled = True
    if wandb_cfg and str(wandb_cfg.get("mode", "")).lower() == "disabled":
        disabled = True
    if mode is not None and str(mode).lower() == "trial":
        disabled = True
    if disabled:
        raise RuntimeError(
            f"Evaluation cannot run on trial-mode runs with WandB disabled (run_id={run_id})."
        )


def main() -> None:
    sns.set_theme(style="whitegrid")
    args = parse_args()
    results_dir = Path(args.results_dir)
    run_ids = json.loads(args.run_ids)
    if not run_ids:
        raise ValueError("run_ids must contain at least one run id")

    cfg = OmegaConf.load(Path("config/config.yaml"))
    entity = cfg.wandb.entity
    project = cfg.wandb.project

    api = wandb.Api()
    summaries: Dict[str, Dict[str, Any]] = {}
    histories: Dict[str, pd.DataFrame] = {}

    for run_id in run_ids:
        try:
            run = api.run(f"{entity}/{project}/{run_id}")
        except Exception as exc:
            print(f"Failed to fetch run {run_id}: {exc}")
            continue
        history = run.history(samples=100000)
        summary = run.summary._json_dict
        config = dict(run.config)
        check_run_is_full(config, run_id)
        histories[run_id] = history
        summaries[run_id] = summary

        run_dir = results_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = run_dir / "metrics.json"
        payload = {
            "history": history.to_dict(orient="records"),
            "summary": summary,
            "config": config,
        }
        save_json(metrics_path, payload)
        print(metrics_path)

        paths: List[Path] = []
        for metric in [
            "accuracy",
            "mean_abs_error",
            "parse_success_rate",
            "state_valid_rate",
            "repair_rate",
            "avg_model_calls",
            "avg_output_tokens",
            "train_loss",
        ]:
            out_path = run_dir / f"{run_id}_learning_curve_{metric}.pdf"
            if plot_line_metric(history, metric, out_path, f"{run_id} {metric}"):
                paths.append(out_path)

        error_hist = run_dir / f"{run_id}_error_histogram.pdf"
        if plot_error_hist(history, error_hist, f"{run_id} Error Histogram"):
            paths.append(error_hist)

        error_box = run_dir / f"{run_id}_abs_error_boxplot.pdf"
        if plot_boxplot(history, "example_abs_error", error_box, f"{run_id} Abs Error"):
            paths.append(error_box)

        parse_bar = run_dir / f"{run_id}_parse_success_bar.pdf"
        if plot_parse_success_bar(history, parse_bar, f"{run_id} Parse Success"):
            paths.append(parse_bar)

        calls_scatter = run_dir / f"{run_id}_calls_vs_correct_scatter.pdf"
        if plot_scatter(history, "example_model_calls", "example_correct", calls_scatter, f"{run_id} Calls vs Correct"):
            paths.append(calls_scatter)

        tokens_box = run_dir / f"{run_id}_output_tokens_boxplot.pdf"
        if plot_boxplot(history, "example_output_tokens", tokens_box, f"{run_id} Output Tokens"):
            paths.append(tokens_box)

        repairs_hist = run_dir / f"{run_id}_repairs_histogram.pdf"
        if plot_histogram(history, "example_repairs_used", repairs_hist, f"{run_id} Repairs Used"):
            paths.append(repairs_hist)

        complexity_path = run_dir / f"{run_id}_complexity_accuracy_bar.pdf"
        if plot_complexity_accuracy(history, complexity_path, f"{run_id} Accuracy by Complexity"):
            paths.append(complexity_path)

        confusion_path = run_dir / f"{run_id}_confusion_matrix_parse_correct.pdf"
        if plot_confusion_matrix(history, confusion_path, f"{run_id} Parse vs Correct"):
            paths.append(confusion_path)

        for path in paths:
            print(path)

    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    aggregated = aggregate_metrics(summaries, PRIMARY_METRIC)
    aggregated_path = comparison_dir / "aggregated_metrics.json"
    save_json(aggregated_path, aggregated)
    print(aggregated_path)

    accuracy_values = {run_id: summaries[run_id].get("accuracy", float("nan")) for run_id in summaries}
    accuracy_plot = comparison_dir / "comparison_accuracy_bar_chart.pdf"
    if plot_comparison_bar(accuracy_values, accuracy_plot, "Accuracy by Run", "accuracy"):
        print(accuracy_plot)

    calls_values = {run_id: summaries[run_id].get("avg_model_calls", float("nan")) for run_id in summaries}
    calls_plot = comparison_dir / "comparison_avg_model_calls_bar_chart.pdf"
    if plot_comparison_bar(calls_values, calls_plot, "Avg Model Calls", "calls"):
        print(calls_plot)

    parse_values = {run_id: summaries[run_id].get("parse_success_rate", float("nan")) for run_id in summaries}
    parse_plot = comparison_dir / "comparison_parse_success_bar_chart.pdf"
    if plot_comparison_bar(parse_values, parse_plot, "Parse Success Rate", "parse_success_rate"):
        print(parse_plot)

    error_values: Dict[str, np.ndarray] = {}
    for run_id, history in histories.items():
        if "example_abs_error" in history.columns:
            vals = history["example_abs_error"].dropna().values
            if len(vals) > 0:
                error_values[run_id] = vals
    error_box = comparison_dir / "comparison_abs_error_boxplot.pdf"
    if plot_comparison_boxplot(error_values, error_box, "Absolute Error by Run", "abs_error"):
        print(error_box)

    table_metrics = [
        "accuracy",
        "mean_abs_error",
        "parse_success_rate",
        "state_valid_rate",
        "repair_rate",
        "avg_model_calls",
        "avg_output_tokens",
    ]
    table_data = {
        metric: [summaries[run_id].get(metric, float("nan")) for run_id in summaries]
        for metric in table_metrics
    }
    metrics_table = pd.DataFrame(table_data, index=list(summaries.keys()))
    table_csv = comparison_dir / "comparison_metrics_table.csv"
    metrics_table.to_csv(table_csv)
    print(table_csv)
    table_pdf = comparison_dir / "comparison_metrics_table.pdf"
    if render_metrics_table(metrics_table, table_pdf, "Comparison Metrics Table"):
        print(table_pdf)

    proposed_ids = [r for r in summaries if "proposed" in r]
    baseline_ids = [r for r in summaries if "comparative" in r or "baseline" in r]
    if proposed_ids and baseline_ids:
        if metric_is_lower_better(PRIMARY_METRIC):
            best_proposed = min(proposed_ids, key=lambda r: summaries[r].get(PRIMARY_METRIC, float("inf")))
            best_baseline = min(baseline_ids, key=lambda r: summaries[r].get(PRIMARY_METRIC, float("inf")))
        else:
            best_proposed = max(proposed_ids, key=lambda r: summaries[r].get(PRIMARY_METRIC, float("-inf")))
            best_baseline = max(baseline_ids, key=lambda r: summaries[r].get(PRIMARY_METRIC, float("-inf")))

        p_hist = histories.get(best_proposed)
        b_hist = histories.get(best_baseline)
        if p_hist is not None and b_hist is not None:
            if "example_correct" in p_hist.columns and "example_correct" in b_hist.columns:
                p_vals = p_hist["example_correct"].dropna().values
                b_vals = b_hist["example_correct"].dropna().values
                if len(p_vals) > 1 and len(b_vals) > 1:
                    stat, p_val = ttest_ind(p_vals, b_vals, equal_var=False)
                    significance = {
                        "best_proposed": best_proposed,
                        "best_baseline": best_baseline,
                        "t_stat": float(stat),
                        "p_value": float(p_val),
                    }
                    sig_path = comparison_dir / "comparison_significance_test.json"
                    save_json(sig_path, significance)
                    print(sig_path)


if __name__ == "__main__":
    main()
