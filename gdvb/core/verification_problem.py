import os
import pathlib
import numpy as np
import toml
import re

from fractions import Fraction as F

from ..artifacts.ACAS import ACAS
from ..artifacts.MNIST import MNIST
from ..artifacts.CIFAR10 import CIFAR10
from ..artifacts.DAVE2 import DAVE2
from ..artifacts.TaxiNet import TaxiNet

from ..nn.layers import Dense, Conv, Transpose, Flatten
from ..pipeline.dispatcher import Task
from ..pipeline.R4V import R4V
from ..pipeline.DNNV import DNNV, DNNV_wb
from ..pipeline.DNNF import DNNF
from ..pipeline.SwarmHost import SwarmHost

from swarm_host.core.problem import VerificationProblem as SHVP


class VerificationProblem:
    def __init__(self, settings, vpc, verification_benchmark):
        self.settings = settings
        self.vpc = vpc
        self.verification_benchmark = verification_benchmark

        self.gen_names()

        self.prop_dir = os.path.join(self.settings.props_dir, self.net_name)
        source_dis_config = open(self.settings.dnn_configs["r4v_config"], "r").read()
        self.distillation_config = toml.loads(source_dis_config)

        self.scale_input = False
        self.training_lost = {}
        self.verification_results = {}

    def gen_names(self):
        self.vp_name = ""
        self.net_name = ""
        for factor, level in [(x, self.vpc[x]) for x in self.vpc]:
            if factor in ["prop"]:
                token = f"{factor}={level}_"
            else:
                token = f"{factor}={level:.{self.settings.precision}f}_"
            self.vp_name += token
            if factor not in ["eps", "prop"]:
                self.net_name += token
        self.vp_name = self.vp_name[:-1]
        self.net_name = self.net_name[:-1]

    def set_distillation_strategies(self, dis_strats):
        drop_ids = []
        added_layers = []
        scale_ids_factors = []
        for ds in dis_strats:
            if ds[0] == "drop":
                drop_ids += [ds[1]]
            elif ds[0] == "add":
                added_layers += [ds[1]]
            elif ds[0] == "scale":
                scale_ids_factors += [[ds[1], ds[2]]]
            elif ds[0] == "scale_input":
                self.scale_input = True
                self.scale_input_factor = ds[1]
            else:
                assert False, "Unknown strategy" + ds

        strategies = [x[0] for x in dis_strats]
        if "scale" in strategies:
            self.net_name += f'_SF={dis_strats[strategies.index("scale")][2]:.{self.settings.precision}f}'
            self.vp_name += f'_SF={dis_strats[strategies.index("scale")][2]:.{self.settings.precision}f}'
        else:
            self.net_name += f"_SF={1:.{self.settings.precision}f}"
            self.vp_name += f"_SF={1:.{self.settings.precision}f}"

        self.drop_ids = drop_ids
        self.scale_ids_factors = scale_ids_factors
        self.added_layers = added_layers

        self.dis_config_path = os.path.join(
            self.settings.dis_config_dir, self.net_name + ".toml"
        )
        self.dis_model_path = os.path.join(
            self.settings.dis_model_dir, self.net_name + ".onnx"
        )
        self.dis_log_path = os.path.join(
            self.settings.dis_log_dir, self.net_name + ".out"
        )
        if self.settings.training_configs["dispatch"]["platform"] == "slurm":
            self.dis_slurm_path = os.path.join(
                self.settings.dis_slurm_dir, self.net_name + ".slurm"
            )
        else:
            self.dis_slurm_path = None

    # calculate neurons/layers
    def calc_order(self, order_by, orig_layers, input_shape):
        if order_by == "nb_neurons":
            self.layers = []
            self.nb_neurons = []
            self.remaining_layer_ids = []
            self.fc_and_conv_kernel_sizes = []

            # create network
            for i in range(len(orig_layers)):
                if i in self.drop_ids:
                    self.layers += [None]
                    self.nb_neurons += [0]
                    self.fc_and_conv_kernel_sizes += [0]
                else:
                    tmp_layers = [x for x in self.layers if x is not None]
                    if not tmp_layers:
                        in_shape = input_shape
                    else:
                        in_shape = tmp_layers[-1].out_shape

                    ol = orig_layers[i]
                    if ol.type == "FC":
                        size = ol.size
                        for x in self.scale_ids_factors:
                            if x[0] == i:
                                size = int(round(size * x[1]))
                                break
                        self.fc_and_conv_kernel_sizes += [size]
                        l = Dense(size, None, None, in_shape)
                        self.nb_neurons += [np.prod(l.out_shape)]
                    elif ol.type == "Conv":
                        size = ol.size
                        for x in self.scale_ids_factors:
                            if x[0] == i:
                                size = int(round(size * x[1]))
                                break
                        self.fc_and_conv_kernel_sizes += [size]
                        l = Conv(
                            size,
                            None,
                            None,
                            ol.kernel_size,
                            ol.stride,
                            ol.padding,
                            in_shape,
                        )
                        self.nb_neurons += [np.prod(l.out_shape)]
                    elif ol.type == "Transpose":
                        l = Transpose(ol.order, in_shape)
                        self.fc_and_conv_kernel_sizes += [0]
                        self.nb_neurons += [0]
                    elif ol.type == "Flatten":
                        l = Flatten(in_shape)
                        self.fc_and_conv_kernel_sizes += [0]
                        self.nb_neurons += [0]
                    else:
                        assert False
                    self.layers += [l]
                    self.remaining_layer_ids += [i]

            # add layers
            for layer in self.added_layers:
                if layer["layer_type"] == "FullyConnected":
                    for layer_id in layer["layer_id"]:
                        in_shape = self.layers[layer_id - 1].out_shape
                        size = layer["parameters"]
                        for x in self.scale_ids_factors:
                            if x[0] == i:
                                size = int(round(size * x[1]))
                                break
                        new_layer = Dense(size, None, None, in_shape)
                        self.layers.insert(layer_id, new_layer)
                        self.nb_neurons.insert(layer_id, np.prod(new_layer.out_shape))
                        self.fc_and_conv_kernel_sizes.insert(layer_id, size)

                elif layer["layer_type"] == "Convolutional":
                    for layer_id in layer["layer_id"]:
                        nb_kernels = layer["parameters"][0]  # number of kernels
                        kernel_size = layer["parameters"][1]
                        stride = layer["parameters"][2]
                        padding = layer["parameters"][3]

                        in_shape = self.layers[layer_id - 1].out_shape
                        for x in self.scale_ids_factors:
                            if x[0] == i:
                                nb_kernels = int(round(nb_kernels * x[1]))
                                break

                        new_layer = Conv(
                            nb_kernels,
                            None,
                            None,
                            kernel_size,
                            stride,
                            padding,
                            in_shape,
                        )
                        self.layers.insert(layer_id, new_layer)
                        self.nb_neurons.insert(layer_id, np.prod(new_layer.out_shape))
                        self.fc_and_conv_kernel_sizes.insert(layer_id, kernel_size)
                else:
                    raise NotImplementedError
            self.layers = [x for x in self.layers if x is not None]
        else:
            assert False

    # override comparators
    def __gt__(self, other):
        if self.order_by == "nb_neurons" and other.order_by == "nb_neurons":
            return np.sum(self.nb_neurons) > np.sum(other.nb_neurons)
        else:
            raise NotImplementedError()

    # writes the R4V training configs
    def write_training_configs(self):
        if "epochs" in self.settings.training_configs:
            self.distillation_config["distillation"]["parameters"][
                "epochs"
            ] = self.settings.training_configs["epochs"]
        training_configs = toml.dumps(self.distillation_config)
        open(self.dis_config_path, "w").write(training_configs)

        # write distillation strategies in order: scale_input > drop > add > scale_layer
        # TODO: use better toml api
        lines = [""]
        if self.scale_input:
            lines += ["[[distillation.strategies.scale_input]]"]
            lines += [f"factor={self.scale_input_factor}\n"]

        if self.drop_ids:
            lines += ["[[distillation.strategies.drop_layer]]"]
            lines += ["layer_id=[" + ", ".join([str(x) for x in self.drop_ids]) + "]"]
            lines += [""]

        if self.added_layers:
            for layer in self.added_layers:
                lines += ["[[distillation.strategies.add_layer]]"]
                lines += [f'layer_type="{layer["layer_type"]}"']
                lines += [f'parameters={layer["parameters"]}']
                lines += [f'activation_function="{layer["activation_function"]}"']
                lines += [f'layer_id={layer["layer_id"]}']
                lines += [""]

        if self.scale_ids_factors:
            lines += ["[[distillation.strategies.scale_layer]]"]
            lines += [
                "layer_id=["
                + ", ".join([str(x[0]) for x in self.scale_ids_factors])
                + "]"
            ]
            lines += [
                "factor=[{}]".format(
                    ", ".join([str(x[1]) for x in self.scale_ids_factors])
                )
            ]
            lines += [""]

        lines += ["[distillation.student]"]
        lines += ['path="' + self.dis_model_path + '"']
        lines = [x + "\n" for x in lines]

        open(self.dis_config_path, "a").writelines(lines)

    # am I trained?
    def trained(self, strict):
        trained = False
        if strict:
            if os.path.exists(self.dis_log_path):
                lines = open(self.dis_log_path, "r").readlines()[-10:]
                for line in lines:
                    if "Process finished successfully" in line:
                        trained = True
                        break
        else:
            trained = os.path.exists(self.dis_log_path)
        return trained

    # train network
    def train(self):
        if not self.settings.override and self.trained(False):
            self.settings.logger.info(f"Skipping trained network ...")
            return
        else:
            self.write_training_configs()

            cmd = R4V(["distill", "debug"]).execute([self.dis_config_path])
            cmds = [cmd]

            exclude = (
                os.environ["train_nodes_exclude"]
                if "train_nodes_exclude" in os.environ
                else None
            )
            dispatch = self.settings.training_configs["dispatch"]
            if exclude:
                dispatch["exclude"] = exclude

            task = Task(
                cmds,
                dispatch,
                "GDVB_T",
                self.dis_log_path,
                self.dis_slurm_path,
                need_warming_up=False,
            )
            self.settings.logger.debug(f"Command: {cmd}")
            task.run()

    def analyze_training(self):
        relative_loss = []
        if os.path.exists(self.dis_log_path):
            lines = open(self.dis_log_path).readlines()
            for line in lines:
                if "validation error" in line:
                    relative_loss += [float(line.strip().split("=")[-1])]
        if len(relative_loss) != self.settings.training_configs["epochs"]:
            self.settings.logger.warning(
                f"Training may not be finished. "
                f"({len(relative_loss)}/{self.settings.training_configs['epochs']}) {self.dis_log_path}"
            )
            ram_issue = False
            CUDA_ram_issue = False
            for l in reversed(lines):
                if "Out of Memory" in l:
                    ram_issue = True
                    break
                if "CUDA out of memory" in l:
                    CUDA_ram_issue = True
                    break
            assert ram_issue or CUDA_ram_issue
            if ram_issue:
                self.settings.logger.error(f"Hardware limit: out of memory.")
            elif CUDA_ram_issue:
                self.settings.logger.error(f"Hardware limit: CUDA out of memory.")
            else:
                raise RuntimeError("Unknown runtime error.")
        return relative_loss

    def gen_prop(self):
        if isinstance(self.verification_benchmark.artifact, ACAS):
            prop_id = self.vpc["prop"]
            self.verification_benchmark.artifact.generate_property(prop_id)

        elif isinstance(
            self.verification_benchmark.artifact, (MNIST, CIFAR10, DAVE2, TaxiNet)
        ):
            data_config = self.distillation_config["distillation"]["data"]
            prop_id = self.vpc["prop"]

            if "eps" in self.vpc:
                eps = F(str(self.vpc["eps"])) * F(
                    str(self.settings.verification_configs["eps"])
                )
            else:
                eps = self.settings.verification_configs["eps"]
            eps = round(float(eps), self.settings.precision)

            skip_layers = (
                0
                if "skip_layers" not in self.settings.verification_configs
                else self.settings.verification_configs["skip_layers"]
            )

            pathlib.Path(self.prop_dir).mkdir(parents=True, exist_ok=True)

            self.verification_benchmark.artifact.generate_property(
                data_config,
                prop_id,
                eps,
                skip_layers,
                self.prop_dir,
                self.settings.seed,
            )
        else:
            raise NotImplementedError

    # am I verified?
    def verified(self):
        log_path = self.veri_log_path
        verified = False
        # self.settings.logger.debug(f"Checking log file: {log_path}")
        if os.path.exists(log_path):
            lines = open(log_path).readlines()
            for line in lines:
                if any(
                    x in line
                    for x in [
                        "(resmonitor) Process finished successfully",
                        "Timeout (terminating process)",
                        "Out of Memory (terminating process)",
                        "Model does not exist",
                    ]
                ):
                    """
                    for x in [
                        # dnnv
                        "  result: ",
                        "Timeout (terminating process)",
                        "Out of Memory (terminating process)",
                        # swarm_host
                        "Result: ",
                        # TODO: fix if possible
                        # hacks for neurify and vnnlib
                        'vnnlib SAT',
                        'vnnlib UNSAT',
                        'vnnlib UNKNOWN'
                    ]
                    """
                    verified = True
                    break
        if not verified:
            self.settings.logger.debug(f"Unverified log file: {log_path}")
        return verified

    # verify network
    def verify(self, tool, options):
        options = [options]
        configs_v = self.settings.verification_configs

        if "debug" in configs_v and configs_v["debug"]:
            options += ["debug"]
        verifier = globals()[tool](options)

        # added 60 for DNNV load time

        time_limit = configs_v["time"]
        memory_limit = configs_v["memory"]

        dnnv_wb_flag = "_wb" if isinstance(verifier, DNNV_wb) else ""
        self.veri_log_path = os.path.join(
            self.settings.veri_log_dir,
            f"{self.vp_name}_T={time_limit}_M={memory_limit}:{verifier.verifier_name}{dnnv_wb_flag}.out",
        )

        if configs_v["dispatch"]["platform"] == "slurm":
            slurm_script_path = os.path.join(
                self.settings.veri_slurm_dir,
                f"{self.vp_name}_T={time_limit}_M={memory_limit}:{verifier.verifier_name}{dnnv_wb_flag}.slurm",
            )
        else:
            slurm_script_path = None

        if not self.settings.override and self.verified():
            self.settings.logger.info("Skipping verified problem ...")
            return

        if "eps" in self.vpc:
            eps = F(str(self.vpc["eps"])) * F(str(configs_v["eps"]))
        else:
            eps = configs_v["eps"]
        eps = round(float(eps), self.settings.precision)

        property_path = os.path.join(
            self.prop_dir, f"robustness_{self.vpc['prop']}_{eps}.py"
        )

        ### Verifier frameworks
        # DNNV family executor
        if any(isinstance(verifier, x) for x in [DNNV, DNNV_wb, DNNF]):
            cmd = f"python -W ignore $DNNV/tools/resmonitor.py -q -T {time_limit+60} -M {memory_limit} "
            cmd += verifier.execute([property_path, "--network N", self.dis_model_path])

        # SwarmHost executor
        elif isinstance(verifier, SwarmHost):
            self.veri_config_path = os.path.join(
                self.settings.veri_config_dir,
                f"{self.vp_name}_T={time_limit}_M={memory_limit}:{verifier.verifier_name}{dnnv_wb_flag}.yaml",
            )
            data_config = self.distillation_config["distillation"]["data"]["transform"][
                "student"
            ]
            p_mean = " ".join(str(x) for x in data_config["mean"])
            p_std = " ".join(str(x) for x in data_config["std"])
            print(f"{data_config}")
            cmd = verifier.execute(
                [
                    f"--onnx {self.dis_model_path}",
                    f"--artifact {self.verification_benchmark.artifact.__name__}",
                    f"--property_id {self.vpc['prop']}",
                    f"--eps {eps}",
                    f"--property_dir {self.prop_dir}",
                    f"--veri_config_path {self.veri_config_path}",
                    f"-t {time_limit}",
                    f"-m {memory_limit}",
                    f"--p_mean {p_mean}",
                    f"--p_std {p_std}",
                    f"--p_mrb"
                    # f"--p_clip",
                ]
            )
        else:
            raise NotImplementedError
        cmds = [cmd]

        nodes = os.environ["verify_nodes"] if "verify_nodes" in os.environ else None
        dispatch = configs_v["dispatch"]
        if nodes:
            dispatch["nodes"] = nodes.split(",")
        warm_up = dispatch["platform"] == "slurm"

        setup_cmds = None if not "setup_cmds" in configs_v else configs_v["setup_cmds"]
        task = Task(
            cmds,
            dispatch,
            "GDVB_V",
            self.veri_log_path,
            slurm_script_path,
            setup_cmds=setup_cmds,
            need_warming_up=warm_up,
        )
        self.settings.logger.debug(f"Command: {cmd}")
        task.run()

    def analyze_verification(self):
        configs_v = self.settings.verification_configs
        verification_results = {}
        verifiers = []
        for tool in configs_v["verifiers"]:
            for options in configs_v["verifiers"][tool]:
                verifier = globals()[tool]([options])
                verifiers += [verifier]

        time_limit = configs_v["time"]
        memory_limit = configs_v["memory"]
        for verifier in verifiers:
            if isinstance(verifier, SwarmHost):
                log_path = os.path.join(
                    self.settings.veri_log_dir,
                    f"{self.vp_name}_T={time_limit}_M={memory_limit}:{verifier.verifier_name}.out",
                )
                vp = SHVP(
                    self.settings.logger,
                    None,
                    options,
                    {"time": configs_v["time"]},
                    {"veri_log_path": log_path},
                )
                verification_answer, verification_time = vp.analyze()

            elif isinstance(verifier, DNNV) or isinstance(verifier, DNNV_wb):
                dnnv_wb_flag = "_wb" if isinstance(verifier, DNNV_wb) else ""
                log_path = os.path.join(
                    self.settings.veri_log_dir,
                    f"{self.vp_name}_T={time_limit}_M={memory_limit}:{verifier.verifier_name}{dnnv_wb_flag}.out",
                )

                if not os.path.exists(log_path):
                    verification_answer = "unrun"
                    self.settings.logger.warning(f"unrun: {log_path}")
                    verification_time = -1
                else:
                    lines = list(reversed(open(log_path, "r").readlines()))

                    if os.path.exists(os.path.splitext(log_path)[0] + ".err"):
                        lines_err = list(
                            reversed(
                                open(
                                    os.path.splitext(log_path)[0] + ".err", "r"
                                ).readlines()
                            )
                        )
                        lines = lines_err + lines

                    verification_answer = None
                    verification_time = None
                    for i, l in enumerate(lines):
                        if re.match(r"INFO*", l):
                            continue

                        if re.search("Timeout", l):
                            verification_answer = "timeout"
                            verification_time = time_limit
                            break

                        if re.search("Out of Memory", l):
                            verification_answer = "memout"
                            for l in lines:
                                if re.search("Duration", l):
                                    verification_time = float(l.split(" ")[9][:-2])
                                    break
                            break

                        # if re.search('RuntimeError: view size is not compatible', l):
                        #    verification_answer = 'error'
                        #    verification_time = time_limit
                        #    break

                        if re.search(" result: ", l):
                            error_patterns = [
                                "PlanetError",
                                "ReluplexError",
                                "ReluplexTranslatorError",
                                "ERANError",
                                "MIPVerifyTranslatorError",
                                "NeurifyError",
                                "NeurifyTranslatorError",
                                "NnenumError",
                                "NnenumTranslatorError",
                                "MarabouError",
                                "VerinetError",
                                "MIPVerifyError",
                            ]
                            if any(re.search(x, l) for x in error_patterns):
                                verification_answer = "error"
                            # elif re.search('Return code: -11', l):
                            #    verification_answer = 'memout'
                            else:
                                verification_answer = l.strip().split(" ")[-1]
                                verification_time = float(
                                    lines[i - 1].strip().split(" ")[-1]
                                )
                            break

                        # Error of neurify
                        if re.search(
                            "ValueError: attempt to get argmax of an empty sequence", l
                        ):
                            verification_answer = "error"
                            verification_time = -1
                            break

                        if re.search("Result: ", l):
                            verification_answer = l.strip().split()[-1]
                            verification_time = float(lines[i - 1].strip().split()[-1])
                            break

                        # exceptions that DNNV didn't catch
                        # exception_patterns = ["Aborted         "]
                        # if any(re.search(x, l) for x in exception_patterns):
                        #    verification_answer = 'exception'
                        #    for l in rlines:
                        #        if re.search('Duration', l):
                        #            verification_time = float(l.split(' ')[9][:-2])
                        #            break
                        #    break

                        # failed jobs that are likely caused by server error
                        rerun_patterns = [
                            "CANCELLED AT",
                            "Unable to open Gurobi license file",
                            "cannot reshape array of size 0 into shape",
                            "property_node = module.body[-1]",
                            "slurmstepd: error: get_exit_code",
                            "Cannot load file containing pickled data",
                            "IndexError: list index out of range",
                            "gurobipy.GurobiError: No Gurobi license",
                            "gurobipy.GurobiError: License expired ",
                            "Cannot allocate memory",
                            "Disk quota exceeded",
                            "ValueError: Unknown arguments: --",
                            "--- Logging error ---",
                            "corrupted size vs. prev_size",
                            "No space left on device",
                        ]
                        if any(re.search(x, l) for x in rerun_patterns):
                            verification_answer = "rerun"
                            verification_time = -1
                            self.settings.logger.warning(
                                f"Failed job({verification_answer}): {log_path}"
                            )
                            break
            else:
                raise NotImplementedError()

            if not verification_answer:
                verification_answer = "undetermined"
                verification_time = -1
                self.settings.logger.warning(f"Undetermined job: {log_path}")

            if False and verification_answer == "error":
                print(f"No!!! Error {log_path}")
                os.remove(log_path)
                os.remove(log_path.replace(".out", ".err"))
                print(f"Removed failed log: {log_path}")

            if False and verification_answer in ["undetermined", "unrun", "rerun"]:
                os.remove(log_path)
                os.remove(log_path.replace(".out", ".err"))
                print(f"Removed failed log: {log_path}")

            assert verification_answer, verification_time
            assert verification_answer in [
                "sat",
                "unsat",
                "unknown",
                "error",
                "timeout",
                "memout",
                "exception",
                "rerun",
                "unrun",
                "undetermined",
                "hardware_limit",
            ], f"{verification_answer}:{log_path}"

            # double check verification time to recover time loss
            if verification_answer in ["sat", "unsat", "unknown"]:
                if verification_time > time_limit:
                    verification_time = time_limit
                    verification_answer = "timeout"

            verification_results[verifier.verifier_name] = [
                verification_answer,
                verification_time,
            ]

        self.verification_results = verification_results
        return verification_results
