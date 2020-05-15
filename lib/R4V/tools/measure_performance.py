#!/usr/bin/env python
import argparse
import glob
import importlib
import numpy as np
import os
import toml
import torch
import torch.nn.functional as F

from collections import defaultdict
from pathlib import Path

from r4v.distillation.config import DataConfiguration
from r4v.distillation.data import get_data_loader
from r4v.nn import load_network


def _parse_args():
    parser = argparse.ArgumentParser(description="Measure the performance of a model.")

    parser.add_argument("model_path", type=str, help="path to model")
    parser.add_argument("data_config", type=str, help="path to the dataset")

    parser.add_argument("--teacher", type=str, help="path to model")

    parser.add_argument("--input_shape", type=int, nargs="+", default=[1, 3, 224, 224])
    parser.add_argument(
        "--input_format", type=str, default="NCHW", choices=["NHWC", "NCHW"]
    )

    parser.add_argument(
        "--teacher_input_shape", type=int, nargs="+", default=[1, 3, 224, 224]
    )
    parser.add_argument(
        "--teacher_input_format", type=str, default="NCHW", choices=["NHWC", "NCHW"]
    )

    parser.add_argument(
        "--loss", default="mse", choices=["accuracy", "mse", "mae", "dronet"]
    )

    parser.add_argument("--cuda", action="store_true")

    parser.add_argument(
        "--plugins", type=str, nargs="+", default=[], help="plugin modules to load"
    )

    return parser.parse_args()


def measure(
    student_model,
    teacher_model,
    device,
    data_loader,
    student_input_shape,
    student_input_format,
    teacher_input_shape,
    teacher_input_format,
    loss_fn_type=None,
):
    student = load_network(
        {
            "model": student_model,
            "input_shape": student_input_shape,
            "input_format": student_input_format,
        }
    ).as_pytorch(maintain_weights=True)
    student.to(device)
    student.eval()

    teacher = None
    if teacher_model is not None:
        teacher = load_network(
            {
                "model": teacher_model,
                "input_shape": teacher_input_shape,
                "input_format": teacher_input_format,
            }
        ).as_pytorch(maintain_weights=True)
        teacher.to(device)
        teacher.eval()

    num_correct = 0.0
    relative_num_correct = 0.0
    student_num_correct = 0.0
    teacher_num_correct = 0.0
    num_samples = 0.0
    num_loss_samples = 0.0
    num_acc_samples = 0.0
    s_loss = 0.0
    t_loss = 0.0
    rel_loss = 0.0
    performance = {}
    student_outputs = torch.Tensor([])
    teacher_outputs = torch.Tensor([])
    targets = torch.Tensor([])
    student_errors = torch.Tensor([])
    teacher_errors = torch.Tensor([])
    for i, (idx, tx, sx, target) in enumerate(data_loader, 1):
        sx = sx.to(device)
        tx = tx.to(device)
        with torch.no_grad():
            sy = student(sx)
            if teacher is not None:
                ty = teacher(tx)
            else:
                ty = sy
        sy = sy.cpu()
        ty = ty.cpu()
        num_samples += target.size(0)
        if loss_fn_type == "accuracy":
            teacher_pred = ty.argmax(-1)
            student_pred = sy.argmax(-1)
            teacher_correct = teacher_pred == target
            student_correct = student_pred == target
            num_correct += student_correct.sum(dtype=torch.float)
            teacher_num_correct += teacher_correct.sum(dtype=torch.float)
            relative_num_correct += (teacher_correct & student_correct).sum(
                dtype=torch.float
            )
            performance = {
                "student_accuracy": (num_correct / num_samples).item(),
                "relative_accuracy": (relative_num_correct / teacher_num_correct).item()
                if teacher_num_correct > 0
                else float("nan"),
                "teacher_accuracy": (teacher_num_correct / num_samples).item(),
            }
            stats = "no stats"
        elif loss_fn_type == "dronet":
            to_degrees = lambda x: x  # * 180 / np.pi
            sy_steer = to_degrees(sy[:, 0])
            sy_coll = sy[:, 1] >= 0.5
            ty_steer = to_degrees(ty[:, 0])
            ty_coll = ty[:, 1] >= 0.5
            target_steer = to_degrees(target[:, 0])
            target_coll = target[:, 1].type(torch.uint8)
            steer_mask = ~np.isnan(target_steer)
            coll_mask = np.isnan(target_steer)

            num_acc_samples += coll_mask.sum().item()
            student_correct = sy_coll[coll_mask] == target_coll[coll_mask]
            teacher_correct = ty_coll[coll_mask] == target_coll[coll_mask]
            student_num_correct += student_correct.sum(dtype=torch.float)
            teacher_num_correct += teacher_correct.sum(dtype=torch.float)
            relative_num_correct += (teacher_correct & student_correct).sum(
                dtype=torch.float
            )

            loss_type = "mse"
            if loss_type == "mae":
                loss_fn = F.l1_loss
            else:
                loss_type = "mse"
                loss_fn = F.mse_loss
            rel_loss += loss_fn(sy_steer, ty_steer).item() * target.size(0)
            num_steer_samples = steer_mask.sum().item()
            if num_steer_samples > 0:
                num_loss_samples += num_steer_samples
                s_loss += (
                    loss_fn(sy_steer[steer_mask], target_steer[steer_mask]).item()
                    * num_steer_samples
                )
                t_loss += (
                    loss_fn(ty_steer[steer_mask], target_steer[steer_mask]).item()
                    * num_steer_samples
                )
            performance = {
                "student_%s" % loss_type: (s_loss / num_loss_samples)
                if num_loss_samples > 0
                else float("nan"),
                "teacher_%s" % loss_type: (t_loss / num_loss_samples)
                if num_loss_samples > 0
                else float("nan"),
                "relative_%s" % loss_type: (rel_loss / num_samples),
                "student_accuracy": (student_num_correct / num_acc_samples).item(),
                "relative_accuracy": (relative_num_correct / teacher_num_correct).item()
                if teacher_num_correct > 0
                else float("nan"),
                "teacher_accuracy": (teacher_num_correct / num_acc_samples).item(),
            }
            stats = "no stats"
            index = np.random.randint(0, len(target))
        else:
            if loss_fn_type == "mse":
                loss_fn = lambda x, y: F.mse_loss(x, y).item()
            elif loss_fn_type == "mae":
                loss_fn = lambda x, y: F.l1_loss(x, y).item()
            else:
                raise ValueError("Unknown loss function type: %s" % loss_fn_type)
            rel_loss += loss_fn(sy, ty) * target.size(0)
            s_loss += loss_fn(sy, target.reshape(sy.shape)) * target.size(0)
            t_loss += loss_fn(ty, target.reshape(ty.shape)) * target.size(0)
            performance = {
                "student_loss": s_loss / num_samples,
                "teacher_loss": t_loss / num_samples,
                "relative_loss": rel_loss / num_samples,
            }
            targets = torch.cat([targets, target])
            student_outputs = torch.cat([student_outputs, sy])
            teacher_outputs = torch.cat([teacher_outputs, ty])
            student_errors = torch.cat(
                [student_errors, (sy - target.reshape(sy.shape))]
            )
            teacher_errors = torch.cat(
                [teacher_errors, (ty - target.reshape(ty.shape))]
            )
            stats = {
                "mean": targets.mean().item(),
                "std": targets.std().item(),
                "student_mean": student_outputs.mean().item(),
                "student_std": student_outputs.std().item(),
                "teacher_mean": teacher_outputs.mean().item(),
                "teacher_std": teacher_outputs.std().item(),
                "student_errors_mean": student_errors.mean().item(),
                "student_errors_std": student_errors.std().item(),
                "teacher_errors_mean": teacher_errors.mean().item(),
                "teacher_errors_std": teacher_errors.std().item(),
            }
            index = np.random.randint(0, len(target))
        if i % 10 == 0:
            print("%7d: %s" % (i * data_loader.batch_size, performance), flush=True)
            print("         %s" % stats)
    print(performance, flush=True)
    print(stats, flush=True)
    return performance


def maybe_int(value):
    try:
        return int(value)
    except:
        return value


def main(args):
    for plugin_path in args.plugins:
        plugin_name = Path(plugin_path).stem
        spec = importlib.util.spec_from_file_location(plugin_name, plugin_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

    config = DataConfiguration(toml.load(args.data_config))
    config.config["_STAGE"] = "val_test"
    data_loader = get_data_loader(config)

    device = torch.device("cpu")
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    print("Using device: %s" % device)

    if os.path.isfile(args.model_path):
        measure(
            args.model_path,
            args.teacher,
            device,
            data_loader,
            args.input_shape,
            args.input_format,
            args.teacher_input_shape,
            args.teacher_input_format,
            args.loss,
        )
    elif os.path.isdir(args.model_path):
        performance_measures = defaultdict(list)
        for model_path in sorted(
            glob.glob(os.path.join(args.model_path, "*.onnx")),
            key=lambda path: tuple(maybe_int(part) for part in path.split(".")),
        ):
            print(model_path)
            model_name = os.path.basename(model_path)
            if not any(
                isinstance(maybe_int(part), int) for part in model_name.split(".")
            ):
                continue
            performance_measure = measure(
                model_path,
                args.teacher,
                device,
                data_loader,
                args.input_shape,
                args.input_format,
                args.teacher_input_shape,
                args.teacher_input_format,
                args.loss,
            )
            measures_seen = set()
            for measure_name in performance_measure.keys():
                measure_name = measure_name.split("_")[-1]
                if measure_name in measures_seen:
                    continue
                measures_seen.add(measure_name)
                performance = {}
                for name, value in performance_measure.items():
                    if not name.endswith(measure_name):
                        continue
                    optimization = "minimum"
                    opt, argopt = np.min, np.argmin
                    if "accuracy" in name:
                        optimization = "maximum"
                        opt, argopt = np.max, np.argmax
                    performance_measures[name].append(value)
                    iteration = len(performance_measures[name]) - 1
                    opt_value = opt(performance_measures[name])
                    opt_iteration = argopt(performance_measures[name])
                    print(
                        "%d - Current iteration with %s %s: %d"
                        % (iteration, optimization, name, opt_iteration)
                    )
                    print(
                        "%d - Current %s %s: %f"
                        % (iteration, optimization, name, opt_value),
                        flush=True,
                    )
                    performance[name] = performance_measures[name]
                plot_loss_vs_epoch(
                    performance,
                    title=args.model_path,
                    path=os.path.join(
                        args.model_path, "performance.%s.tmp.png" % measure_name
                    ),
                )
        print()
        print("Best Performing Iterations")
        print("==========================")
        measures_seen = set()
        for measure_name in performance_measure.keys():
            measure_name = measure_name.split("_")[-1]
            if measure_name in measures_seen:
                continue
            measures_seen.add(measure_name)
            performance = {}
            for name, value in performance_measure.items():
                if not name.endswith(measure_name):
                    continue
                optimization = "minimum"
                opt, argopt = np.min, np.argmin
                if "accuracy" in name:
                    optimization = "maximum"
                    opt, argopt = np.max, np.argmax
                performance_measures[name].append(value)
                opt_value = opt(performance_measures[name])
                opt_iteration = argopt(performance_measures[name])
                print("Iteration with %s %s: %d" % (optimization, name, opt_iteration))
                optimization = optimization[0].upper() + optimization[1:]
                print("%s %s: %f\n" % (optimization, name, opt_value), flush=True)
                performance[name] = performance_measures[name]
            plot_loss_vs_epoch(
                performance,
                title=args.model_path,
                path=os.path.join(args.model_path, "performance.%s.png" % measure_name),
            )
    else:
        print("%s does not exist." % args.model_path)


if __name__ == "__main__":
    main(_parse_args())
