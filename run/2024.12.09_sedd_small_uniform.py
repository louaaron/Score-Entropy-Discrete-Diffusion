import os
import argparse
import pathlib
import shutil
from datetime import datetime
from subprocess import Popen
import numpy as np


class Overrides(object):
    def __init__(self):
        self.kvs = dict()

    def add(self, key, values):
        value = ",".join(str(v) for v in values)
        assert key not in self.kvs
        self.kvs[key] = value

    def cmd(self):
        cmd = []
        for k, v in self.kvs.items():
            cmd.append(f"{k}={v}")
        return cmd


def make_code_snap(experiment):
    now = datetime.now()
    snap_dir = pathlib.Path("/checkpoint/storygen/zqq/sedd/exp/")
    snap_dir /= now.strftime("%Y.%m.%d")
    snap_dir /= now.strftime("%H%M%S") + f"_{experiment}"
    snap_dir.mkdir(exist_ok=True, parents=True)

    def copy_dir(src_dir, dst_dir, pat):
        for f in src_dir.glob(pat):
            shutil.copy(f, dst_dir / f.name)

    src_dir = pathlib.Path("/home/zqq/sedd/")
    dst_dir = snap_dir / "code"
    dst_dir.mkdir(exist_ok=True, parents=True)
    copy_dir(src_dir, dst_dir, "*.py")

    dirs_to_rcopy = ["configs", "model"]
    for dd in dirs_to_rcopy:
        to_copy = src_dir / dd
        os.system(f"cp -rf {to_copy} {dst_dir}")

    return snap_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str)
    # parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--devlab", action="store_true")
    args = parser.parse_args()

    overrides = Overrides()
    if not args.local:
        overrides.add("hydra/launcher", ["submitit_slurm"])
        if args.devlab:
            overrides.add("hydra.launcher.partition", ["devlab"])

    snap_dir = make_code_snap(args.experiment)
    print(str(snap_dir))

    overrides.add("hydra.sweep.dir", [str(snap_dir)])
    overrides.add("hydra.launcher.submitit_folder", [str(snap_dir / "slurm")])
    overrides.add("experiment", [args.experiment])

    """
    ngpus                     the number of gpus to use in training (using pytorch DDP)
    training.accum            number of accumulation steps, set to 1 for small and 2 for medium (assuming an 8x80GB node)
    noise.type                one of geometric, loglinear
    graph.type                one of uniform, absorb
    model                     one of small, medium
    model.scale_by_sigma      set to False if graph.type=uniform (not yet configured)
    """
    overrides.add("ngpus", values=[8],)
    overrides.add("training.accum", values=[1])
    overrides.add("noise.type", values=["geometric"])
    overrides.add("graph.type", values=["uniform"])
    overrides.add("model", values=["small"])
    overrides.add("model.scale_by_sigma", values=[False])


    cmd = ["python", str(snap_dir / "code" / "train.py"), "-m"]
    cmd += overrides.cmd()

    if args.dry:
        print(" ".join(cmd))
    else:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(snap_dir / "code")
        env["PYOPENGL_PLATFORM"] = ""
        p = Popen(cmd, env=env)
        p.communicate()


if __name__ == "__main__":
    main()
