#!/usr/bin/env python
import argparse
import contextlib
import os
import pandas as pd
import select
import shlex
import subprocess as sp
import time

from pathlib import Path


def memory_t(value):
    if isinstance(value, int):
        return value
    elif value.lower().endswith("g"):
        return int(value[:-1]) * 1_000_000_000
    elif value.lower().endswith("m"):
        return int(value[:-1]) * 1_000_000
    elif value.lower().endswith("k"):
        return int(value[:-1]) * 1000
    else:
        return int(value)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_csv", type=Path)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("property_csv", type=Path)
    parser.add_argument("verifier", choices=["eran", "neurify", "planet", "reluplex", "mipverify"])

    parser.add_argument(
        "-n",
        "--ntasks",
        type=int,
        default=float("inf"),
        help="The max number of running verification tasks.",
    )

    parser.add_argument(
        "-T", "--time", default=-1, type=float, help="The max running time in seconds."
    )
    parser.add_argument(
        "-M",
        "--memory",
        default=-1,
        type=memory_t,
        help="The max allowed memory in bytes.",
    )
    return parser.parse_args()


@contextlib.contextmanager
def lock(filename: Path, *args, **kwargs):
    lock_filename = filename.with_suffix(".lock")
    try:
        while True:
            try:
                lock_fd = os.open(lock_filename, os.O_CREAT | os.O_WRONLY | os.O_EXCL)
                break
            except IOError as e:
                pass
        yield
    finally:
        os.close(lock_fd)
        os.remove(lock_filename)


def wait(pool, timeout=float("inf")):
    start_t = time.time()
    while time.time() - start_t < timeout:
        for index, task in enumerate(pool):
            while select.select([task.stderr], [], [], 0)[0]:
                stderr_line = task.stderr.readline()
                if stderr_line == "":
                    break
                task.stderr_lines.append(stderr_line)
                print(f"{{{task.network}.{task.prop} (STDERR)}}: {stderr_line.strip()}")
            if task.poll() is not None:
                task.stdout_lines.extend(task.stdout.readlines())
                task.stderr_lines.extend(task.stderr.readlines())
                return pool.pop(index)
    for index, task in enumerate(pool):
        if task.poll() is not None:
            task.stdout_lines.extend(task.stdout.readlines())
            task.stderr_lines.extend(task.stderr.readlines())
            return pool.pop(index)
    raise RuntimeError("Timeout while waiting for task completion.")


def parse_verification_output(stdout_lines, stderr_lines):
    time = None
    result_line = stderr_lines[-1]
    if "finished successfully" in result_line:
        try:
            result_lines = []
            at_result = False
            for line in stdout_lines:
                if "dnnv.verifiers" in line:
                    at_result = True
                elif at_result and ("  result:" in line) or ("  time:" in line):
                    result_lines.append(line.strip())
            print("DEBUG<<", stdout_lines)
            result = result_lines[0].split(maxsplit=1)[-1]
            time = float(result_lines[1].split()[-1])
        except Exception as e:
            result = f"error({type(e).__name__})"
            raise e
    elif "Out of Memory" in result_line:
        result = "outofmemory"
        time = float(stderr_lines[-2].split()[-3][:-2])
    elif "Timeout" in result_line:
        result = "timeout"
        time = float(stderr_lines[-2].split()[-3][:-2])
    else:
        result = "!"
    print("  result:", result)
    print("  time:", time)
    return result, time


def update_results(results_csv, network, prop, result, time):
    with lock(results_csv):
        df = pd.read_csv(results_csv)
        df.at[(df["Network"] == network) & (df["Property"] == prop), "Result"] = result
        df.at[(df["Network"] == network) & (df["Property"] == prop), "Time"] = time
        df.to_csv(results_csv, index=False)


def main(args):
    with lock(args.results_csv):
        if not args.results_csv.exists():
            with open(args.results_csv, "w+") as f:
                f.write("Network,Property,Result,Time\n")
    network_property_pairs = set()
    property_df = pd.read_csv(args.property_csv)
    for network in args.model_dir.iterdir():
        for prop in property_df["id"]:
            network_property_pairs.add((network.name, prop))

    pool = []
    while len(network_property_pairs) > 0:
        with lock(args.results_csv):
            df = pd.read_csv(args.results_csv)
            for row in df[["Network", "Property"]].itertuples():
                network = row.Network
                prop = row.Property
                network_property_pairs.discard((network, prop))
            if len(network_property_pairs) == 0:
                break
            network, prop = network_property_pairs.pop()
            df = df.append({"Network": network, "Property": prop}, ignore_index=True)
            df.to_csv(args.results_csv, index=False)

        property_filename = property_df[property_df["id"] == prop][
            "property_filename"
        ].values.item()
        resmonitor = "python ./tools/resmonitor.py"
        resmonitor_args = f"{resmonitor} -M {args.memory} -T {args.time}"
        verifier_args = f"python -m dnnv {args.model_dir / network} {property_filename} --{args.verifier}"
        # slurm_args = "srun --exclusive -n1 " if os.environ.get("SLURM_JOB_NAME") else ""
        slurm_args = ""
        run_args = f"{slurm_args}{resmonitor_args} {verifier_args}"
        print(run_args)

        proc = sp.Popen(
            shlex.split(run_args), stdout=sp.PIPE, stderr=sp.PIPE, encoding="utf8"
        )
        proc.network = network
        proc.prop = prop
        proc.stdout_lines = []
        proc.stderr_lines = []
        pool.append(proc)

        while len(pool) >= args.ntasks:
            finished_task = wait(pool, timeout=2 * args.time)
            print("FINISHED:", " ".join(proc.args))
            result, time = parse_verification_output(
                finished_task.stdout_lines, finished_task.stderr_lines
            )
            update_results(
                args.results_csv,
                finished_task.network,
                finished_task.prop,
                result,
                time,
            )
    while len(pool):
        finished_task = wait(pool, timeout=2 * args.time)
        print("FINISHED:", " ".join(proc.args))
        result, time = parse_verification_output(
            finished_task.stdout_lines, finished_task.stderr_lines
        )
        update_results(
            args.results_csv, finished_task.network, finished_task.prop, result, time
        )


if __name__ == "__main__":
    main(_parse_args())
