import contextlib
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

from ml_lab.experiments import (FEATURES, Dataset, benchmark, clean_zeros,
                                load_dataset, make_pipeline, split_indices)
from ml_lab.reporting import collect_reports
import run


class DatasetTests(unittest.TestCase):
    def csv_file(self, rows=None, headers=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "data.csv"
        if rows is None:
            rows = [[1, 100, 70, 20, 80, 25, .5, 30, i % 2] for i in range(60)]
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(headers or FEATURES + ["Outcome"])
            writer.writerows(rows)
        return path

    def test_cleaning_is_stateless_and_preserves_valid_zeros(self):
        X = np.zeros((2, 8))
        result = clean_zeros(X)
        self.assertTrue(np.isnan(result[:, [1, 2, 3, 4, 5]]).all())
        self.assertTrue((result[:, [0, 6, 7]] == 0).all())
        self.assertTrue((X == 0).all())

    def test_cleaning_rejects_wrong_feature_count(self):
        with self.assertRaises(ValueError):
            clean_zeros(np.ones((4, 7)))

    def test_csv_valid_and_missing_feature(self):
        rows = [[1, "", 70, 20, 80, 25, .5, 30, i % 2] for i in range(60)]
        result = load_dataset("diabetes", self.csv_file(rows))
        self.assertEqual(result.X.shape, (60, 8))
        self.assertTrue(np.isnan(result.X[:, 1]).all())
        self.assertEqual(result.features, FEATURES)

    def test_headers_exact_and_unique(self):
        for header in [FEATURES + ["wrong"], ["Age"] * 8 + ["Outcome"],
                       list(reversed(FEATURES)) + ["Outcome"]]:
            with self.subTest(header=header), self.assertRaises(ValueError):
                load_dataset("diabetes", self.csv_file(headers=header))

    def test_invalid_width(self):
        with self.assertRaisesRegex(ValueError, "nine fields"):
            load_dataset("diabetes", self.csv_file([[1, 2]]))

    def test_nonfinite_or_invalid_target_rejected(self):
        for value in ("", "nan", "inf", ".5", "not-a-number"):
            rows = [[1, 100, 70, 20, 80, 25, .5, 30, i % 2] for i in range(60)]
            rows[0][-1] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                load_dataset("diabetes", self.csv_file(rows))

    def test_negative_feature_rejected(self):
        rows = [[-1, 100, 70, 20, 80, 25, .5, 30, i % 2] for i in range(60)]
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            load_dataset("diabetes", self.csv_file(rows))

    def test_feature_infinity_rejected(self):
        rows = [[1, "inf", 70, 20, 80, 25, .5, 30, i % 2] for i in range(60)]
        with self.assertRaises(ValueError):
            load_dataset("diabetes", self.csv_file(rows))

    def test_too_few_per_class_rejected(self):
        with self.assertRaisesRegex(ValueError, "20 rows"):
            load_dataset("diabetes", self.csv_file([[1, 100, 70, 20, 80, 25, .5, 30, i % 2] for i in range(20)]))

    def test_quick_subset_reproducible_and_stratified(self):
        a, b = load_dataset("digits", quick=True), load_dataset("digits", quick=True)
        np.testing.assert_array_equal(a.X, b.X)
        self.assertEqual(len(a.y), 400)
        self.assertEqual(len(np.unique(a.y)), 10)

    def test_quick_subset_rejects_extreme_class_imbalance(self):
        rows = [[1, 100, 70, 20, 80, 25, .5, 30, int(i < 20)] for i in range(1000)]
        with self.assertRaisesRegex(ValueError, "without --quick"):
            load_dataset("diabetes", self.csv_file(rows), quick=True)

    def test_split_disjoint_complete_and_reproducible(self):
        y = load_dataset("synthetic").y
        splits = split_indices(y)
        self.assertEqual(len(set(np.concatenate(splits))), len(y))
        for i, group in enumerate(splits):
            np.testing.assert_array_equal(group, split_indices(y)[i])
            self.assertEqual(set(y[group]), {0, 1})
            for other in splits[i + 1:]:
                self.assertFalse(set(group) & set(other))


class LeakageTests(unittest.TestCase):
    def test_imputer_and_scaler_refit_on_each_raw_cv_fold(self):
        X = np.arange(120 * 8, dtype=float).reshape(120, 8) + 1
        X[::9, 1:3] = 0
        X[-1, :] = 1e8
        y = np.tile([0, 1], 60)
        splitter = StratifiedKFold(3, shuffle=True, random_state=42)
        pipeline = make_pipeline(DummyClassifier(), zero_missing=True, pca=.95)
        evaluated = cross_validate(pipeline, X, y, cv=splitter,
                                   return_estimator=True, error_score="raise")
        for fitted, (train, _) in zip(evaluated["estimator"], splitter.split(X, y)):
            raw = clean_zeros(X[train])
            np.testing.assert_allclose(fitted["imputer"].statistics_, np.nanmedian(raw, axis=0))
            imputed = fitted["imputer"].transform(raw)
            np.testing.assert_allclose(fitted["scaler"].mean_, imputed.mean(axis=0))
            self.assertEqual(fitted["scaler"].n_samples_seen_, len(train))
            self.assertEqual(fitted["pca"].n_samples_, len(train))

    def test_empty_column_keeps_feature_dimensions(self):
        X = np.ones((20, 8))
        X[:, 1] = np.nan
        pipeline = make_pipeline(DummyClassifier(), zero_missing=True)
        pipeline.fit(X, np.tile([0, 1], 10))
        self.assertEqual(pipeline[:-1].transform(X).shape, (20, 8))

    def test_test_labels_cannot_choose_model(self):
        data = load_dataset("synthetic")
        splits = split_indices(data.y)
        alternative = Dataset(data.name, data.X.copy(), data.y.copy(), data.features.copy())
        alternative.y[splits[2]] = 1 - alternative.y[splits[2]]
        configs = {"Linear": (LogisticRegression(max_iter=500), {"model__C": [1.]}),
                   "Dummy": (DummyClassifier(), {"model__strategy": ["most_frequent"]})}
        with patch("ml_lab.experiments.model_configs", return_value=configs), \
             patch("ml_lab.experiments.split_indices", return_value=splits):
            first, second = benchmark(data, quick=True), benchmark(alternative, quick=True)
        self.assertEqual(first["selected_model"], second["selected_model"])
        for a, b in zip(first["candidates"], second["candidates"]):
            for key in ("cv_f1_mean", "validation_f1", "validation_accuracy", "params"):
                self.assertEqual(a[key], b[key])

    def test_final_pipeline_refits_development_excluding_test_outlier(self):
        data = load_dataset("synthetic")
        splits = split_indices(data.y)
        data.X[splits[2], :] = 1e9
        actual_medians = []
        from ml_lab.experiments import test_metrics as original
        def inspect(model, X, y):
            actual_medians.append(model["imputer"].statistics_.copy())
            return original(model, X, y)
        configs = {"Linear": (LogisticRegression(max_iter=500), {"model__C": [1.]})}
        with patch("ml_lab.experiments.model_configs", return_value=configs), \
             patch("ml_lab.experiments.test_metrics", side_effect=inspect):
            result = benchmark(data, quick=True)
        expected = np.median(data.X[np.concatenate(splits[:2])], axis=0)
        self.assertEqual(len(actual_medians), 2)  # selected model + baseline only
        for median in actual_medians:
            np.testing.assert_allclose(median, expected)
        self.assertEqual(sum(result["split_sizes"].values()), len(data.y))


class ReportingTests(unittest.TestCase):
    def report(self, value=.8, **updates):
        result = {"schema_version": 1, "kind": "deep", "dataset": "synthetic-images", "mode": "smoke",
                  "results": [{"model": "CNN", "test_accuracy": value}]}
        result.update(updates)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "report.json"
        path.write_text(json.dumps(result), encoding="utf-8")
        return path

    def test_missing_experiments_are_not_filled_with_zeros(self):
        self.assertEqual(collect_reports([]), [])

    def test_mode_and_stage_retained(self):
        rows = collect_reports([self.report()])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["mode"], "smoke")
        self.assertEqual(rows[0]["stage"], "test")
        self.assertEqual(rows[0]["dataset"], "synthetic-images")

    def test_invalid_metric_rejected(self):
        for value in (float("nan"), float("inf"), -1, 1.01, "0.8", True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                collect_reports([self.report(value)])

    def test_unknown_schema_rejected(self):
        for update in ({"schema_version": 99}, {"kind": "unknown"}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                collect_reports([self.report(**update)])


class CliTests(unittest.TestCase):
    def test_invalid_seed_and_csv_combination(self):
        for options in (["--seed", "-1"], ["--seed", str(2**32)], ["--dataset", "digits", "--csv", "x.csv"]):
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                run.main(options)

    def test_existing_output_refused_before_training(self):
        with tempfile.TemporaryDirectory() as directory, patch("run.benchmark") as train:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                run.main(["--output", directory])
            train.assert_not_called()

    def test_cli_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "new"
            environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
            result = subprocess.run([sys.executable, "run.py", "--dataset", "synthetic",
                                     "--quick", "--output", str(target)],
                                    capture_output=True, text=True, env=environment, timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((target / "report.json").read_text())
            self.assertEqual(report["mode"], "quick")
            self.assertEqual(report["samples"], 300)
            self.assertEqual(len(report["candidates"]), 4)
            self.assertTrue((target / "development.csv").is_file())
            self.assertIn(report["selected_model"], (target / "README.md").read_text())


class NotebookTests(unittest.TestCase):
    def test_all_cells_compile_without_colab_or_shell_magic(self):
        paths = list(Path(".").glob("Task*.ipynb"))
        self.assertEqual(len(paths), 6)
        for path in paths:
            notebook = json.loads(path.read_text(encoding="utf-8"))
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] == "code":
                    source = cell["source"]
                    source = "".join(source) if isinstance(source, list) else source
                    compile(source, f"{path.name}:{index}", "exec")
                    self.assertFalse(cell["outputs"])
                    self.assertIsNone(cell["execution_count"])


if __name__ == "__main__":
    unittest.main()
