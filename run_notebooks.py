"""Execute notebook code cells in order, without IPython magic or hidden state."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def execute(path):
    # Each notebook gets an isolated process and a fresh global namespace.
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__", "__file__": str(path)}
    import matplotlib.pyplot as plt
    plt.show = lambda *args, **kwargs: None  # runner writes figures without interactive display
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = cell["source"]
        if isinstance(source, list):
            source = "".join(source)
        print(f"{path.name}: cell {index}", flush=True)
        exec(compile(source, f"{path.name}:cell-{index}", "exec"), namespace)
        # plt.show is noninteractive under the runner's Agg backend.
        import matplotlib.pyplot as plt
        plt.close("all")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", type=int, default=[1, 2, 3, 6])
    parser.add_argument("--smoke", action="store_true", help="Reduced data/epochs; synthetic deep-learning inputs")
    parser.add_argument("--execute", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    if args.execute:
        path = args.execute.resolve()
        if path.parent != ROOT or path.name not in {p.name for p in ROOT.glob("Task*.ipynb")}:
            parser.error("Only the six repository notebooks can be executed")
        execute(path)
        return
    if len(args.tasks) != len(set(args.tasks)) or any(t not in range(1, 7) for t in args.tasks):
        parser.error("tasks must be distinct numbers from 1 to 6")
    environment = dict(os.environ, MPLBACKEND="Agg", ML_LAB_SMOKE="1" if args.smoke else "0",
                       OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       PYTHONIOENCODING="utf-8", TF_NUM_INTEROP_THREADS="1", TF_NUM_INTRAOP_THREADS="1", TF_CPP_MIN_LOG_LEVEL="2")
    for task in args.tasks:
        paths = list(ROOT.glob(f"Task{task}_*.ipynb"))
        if len(paths) != 1:
            raise RuntimeError(f"Expected one notebook for task {task}")
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--execute", str(paths[0])],
                       cwd=ROOT, env=environment, check=True, timeout=900 if args.smoke else None)
    print("Notebook execution completed. Figures and results are local generated files.")


if __name__ == "__main__":
    main()
