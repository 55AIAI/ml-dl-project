"""Keep test, validation and CV metrics separate when collecting notebook reports."""
import json
import math
from pathlib import Path


def collect_reports(paths):
    rows = []
    for path in paths:
        path = Path(path)
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("schema_version") != 1 or report.get("kind") not in ("classical", "pca", "deep"):
            raise ValueError(f"Unsupported report schema: {path.name}")
        base = {"source": path.name, "dataset": report["dataset"], "mode": report["mode"]}

        def add(model, protocol, stage, metric, value):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Nonnumeric metric: {path.name}")
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Metric must be finite and in [0, 1]: {path.name}")
            rows.append({**base, "model": model, "protocol": protocol,
                         "stage": stage, "metric": metric, "value": value})

        if report["kind"] == "classical":
            for candidate in report["candidates"]:
                add(candidate["model"], "tuned_on_train", "cv_tuning", "weighted_f1", candidate["cv_f1_mean"])
                add(candidate["model"], "fitted_on_train", "validation", "weighted_f1", candidate["validation_f1"])
            add(report["selected_model"], "selected_then_refit_train_validation", "test",
                "accuracy", report["test"]["accuracy"])
            add("Most frequent baseline", "fitted_train_validation", "test",
                "accuracy", report["baseline_test"]["accuracy"])
        elif report["kind"] == "pca":
            for result in report["results"]:
                add("Random Forest + " + result["transform"], "training_partition_only", "cv",
                    "accuracy", result["cv_accuracy_mean"])
        else:
            for result in report["results"]:
                add(result["model"], "predeclared_model_train_only", "test",
                    "accuracy", result["test_accuracy"])
    return rows
