import os
import subprocess
import uuid
import time
import re
import tempfile


class Task:
    def __init__(
        self,
        cmds,
        dispatch,
        task_name,
        output_path,
        slurm_path,
        setup_cmds=None,
        need_warming_up=False,
    ):
        self.cmds = cmds
        self.platform = dispatch["platform"]

        self.output_path = output_path
        # self.error_path = f"{os.path.splitext(output_path)[0]}.err"
        self.error_path = output_path

        self.exclude = dispatch["exclude"] if "exclude" in dispatch else None
        self.partition = dispatch["partition"] if "partition" in dispatch else None

        self.reservation = (
            dispatch["reservation"] if "reservation" in dispatch else None
        )
        self.nb_cpus = dispatch["nb_cpus"] if "nb_cpus" in dispatch else 1
        self.nodes = dispatch["nodes"] if "nodes" in dispatch else None
        self.task_per_node = (
            dispatch["task_per_node"] if "task_per_node" in dispatch else None
        )
        self.setup_cmds = setup_cmds
        self.need_warming_up = need_warming_up
        if self.platform == "slurm":
            self.slurm_path = slurm_path
            nb_gpus = dispatch["nb_gpus"] if "nb_gpus" in dispatch else 0
            self.configure_slurm(self.cmds, task_name, nb_gpus)

    # execute task
    def run(self):
        if self.platform == "local":
            if self.setup_cmds:
                self.cmds = self.setup_cmds + self.cmds
            for cmd in self.cmds:
                cmd += f" > {self.output_path} 2> {self.error_path}"
                subprocess.run(cmd, shell=True)

        elif self.platform == "slurm":
            cmd = "sbatch"
            cmd += f" --reservation {self.reservation}" if self.reservation else ""

            # if self.nodes:
            # print(f"Requesting a node from: {self.nodes}")
            # node = self.request_node()
            # cmd += f" -w {node}"
            cmd += f" {self.slurm_path}"
            print(cmd)
            subprocess.run(cmd, shell=True)

        else:
            assert False

    # SLURM SCRIPTS
    # TODO: make more general
    # configure slurm script if running on slurm server
    def configure_slurm(self, cmds, task_name, nb_gpus):
        tmpdir = f"/tmp/{uuid.uuid1()}"
        lines = [
            "#!/bin/sh",
            f"#SBATCH --job-name={task_name}",
            f"#SBATCH --output={self.output_path}",
            f"#SBATCH --error={self.error_path}",
            f"#SBATCH --ntasks={self.nb_cpus}",
        ]

        if self.exclude:
            lines += [f"#SBATCH --exclude={self.exclude}"]
        if self.partition:
            lines += [f"#SBATCH --partition={self.partition}"]
        if nb_gpus != 0:
            lines += [f"#SBATCH --gres=gpu:{nb_gpus}"]
        lines += [
            f"export TMPDIR={tmpdir}",
            f"mkdir {tmpdir}",
            "cat /proc/sys/kernel/hostname",
            f"export GRB_LICENSE_FILE={os.environ['GRB_LICENSE_FILE']}",
        ]
        if self.setup_cmds:
            lines += self.setup_cmds
        if self.need_warming_up:
            lines += ['echo "********Dry_Run********"']
            lines += cmds
        lines += ['echo "********Wet_Run********"']
        lines += cmds
        lines += [f"rm -rf {tmpdir}"]

        lines = [x + "\n" for x in lines]
        open(self.slurm_path, "w").writelines(lines)

    def request_node(self):
        while True:
            sq_file = tempfile.NamedTemporaryFile().name
            si_file = tempfile.NamedTemporaryFile().name

            sq_cmd = f"squeue  > {sq_file}"
            os.system(sq_cmd)
            sq_lines = open(sq_file, "r").readlines()[1:]
            os.remove(sq_file)

            nodes_alive = {}
            for node in self.nodes:
                nodes_alive[node] = 0

            alive = True
            for l in sq_lines:
                if "ReqNodeNotAvail" in l and "V_" in l:
                    unavil_node = l[:-1].split(",")[-1].split(":")[1][:-1]
                    if unavil_node in self.nodes:
                        alive = False
                        cmd = f"sinfo > {si_file}"
                        os.system(cmd)
                        si_lines = open(si_file, "r").readlines()
                        os.remove(si_file)

                        temp = re.compile("([a-zA-Z]+)([0-9]+)")
                        name, digits = temp.match(unavil_node).groups()
                        for l in si_lines:
                            if "drain" in l or "down" in l or "drng" in l:
                                if name in l and digits in l:
                                    self.nodes.remove(unavil_node)
                                    if len(self.nodes) == 0:
                                        print("No available nodes. exiting.")
                                        exit()
                                    alive = True
                    if alive:
                        break

                if " R " in l and l != "":
                    node = l.strip().split(" ")[-1]
                    if node in self.nodes:
                        nodes_alive[node] += 1

            if not alive:
                print("Nodes unavailable. Waiting ...")
                time.sleep(10)
                continue

            goon = False
            for na in nodes_alive:
                if nodes_alive[na] < self.task_per_node:
                    goon = True
                    break

            if goon:
                time.sleep(1)
                break

        return na
