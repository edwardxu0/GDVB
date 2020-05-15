"""
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .cli import add_subparser
from .config import DistillationConfiguration
from .data import get_data_loader
from .. import logging
from ..nn import load_network


class DistillationError(RuntimeError):
    pass


def _check_smaller(teacher, student, device=torch.device("cpu")):
    logger = logging.getLogger(__name__)
    num_neurons_teacher = teacher.num_neurons(device)
    num_neurons_student = student.num_neurons(device)

    logger.info(
        "number of neurons: teacher=%d student=%d ratio=%f",
        num_neurons_teacher,
        num_neurons_student,
        num_neurons_student / num_neurons_teacher,
    )
    fewer_neurons = True
    if num_neurons_student > num_neurons_teacher:
        logger.warning(
            "student has more neurons than the teacher: teacher=%d student=%d ratio=%f",
            num_neurons_teacher,
            num_neurons_student,
            num_neurons_student / num_neurons_teacher,
        )
        fewer_neurons = False

    num_params_teacher = teacher.num_parameters
    num_params_student = student.num_parameters

    logger.info(
        "number of parameters: teacher=%d student=%d ratio=%f",
        num_params_teacher,
        num_params_student,
        num_params_student / num_params_teacher,
    )
    fewer_parameters = True
    if num_params_student > num_params_teacher:
        logger.warning(
            "student has more parameters than the teacher: teacher=%d student=%d ratio=%f",
            num_params_teacher,
            num_params_student,
            num_params_student / num_params_teacher,
        )
        fewer_parameters = False

    return fewer_neurons and fewer_parameters


def get_device(config):
    logger = logging.getLogger(__name__)
    if config.get("cuda", False):
        if torch.cuda.is_available():
            logger.debug("Cuda is available. Default device: %s", torch.device("cuda"))
            return torch.device("cuda")
        else:
            logger.warning("cuda specified in config, but is not available")
    return torch.device("cpu")


def get_loss_function(loss, params):
    if loss == "kd":
        alpha = params.get("alpha", 1.0)
        T = params.get("T", 1.0)

        def loss_fn(ty, sy, y):
            soft_ty = F.softmax(ty / T, dim=-1)
            log_soft_sy = F.log_softmax(sy / T, dim=-1)
            return F.kl_div(
                log_soft_sy, soft_ty, reduction="batchmean"
            ) * alpha * T * T + F.nll_loss(log_soft_sy, y) * (1.0 - alpha)

        return loss_fn

    elif loss == "mse":
        criterion = torch.nn.MSELoss(reduction="sum")
        return lambda ty, sy, y=None: criterion(sy, ty)
    elif loss == "balanced_mse":
        window_size = params.get("window_size", 10000)
        epsilon = params.get("epsilon", 1e-6)

        class BalancedMSELoss:
            def __init__(self, window_size=1000, epsilon=1e-6):
                self.window = torch.Tensor()
                self.window_size = window_size
                self.epsilon = epsilon

            def __call__(self, ty, sy, y=None):
                if y is None:
                    mean = self.window.mean(dim=0)
                    std = self.window.std(dim=0)
                    sy_ = (sy - mean) / (std + self.epsilon)
                    ty_ = (ty - mean) / (std + self.epsilon)
                    return F.mse_loss(sy_, ty_, reduction="sum")
                self.window = self.window.to(ty.device)
                self.window = torch.cat([self.window, ty])[-self.window_size :]
                if len(self.window) == 1:
                    mean = self.window[0]
                    std = self.window[0]
                else:
                    mean = self.window.mean(dim=0)
                    std = self.window.std(dim=0)
                sy_ = (sy - mean) / (std + self.epsilon)
                ty_ = (ty - mean) / (std + self.epsilon)
                loss = F.mse_loss(sy_, ty_, reduction="sum")
                return loss

        return BalancedMSELoss(window_size=window_size, epsilon=epsilon)

    else:
        raise ValueError("Unknown loss type: %s" % loss)


def get_optimizer(optimization_algorithm, dnn, params):
    if optimization_algorithm == "sgd":
        return optim.SGD(
            dnn.parameters(),
            lr=params.get("learning_rate", 0.01),
            momentum=params.get("momentum", 0),
            weight_decay=params.get("weight_decay", 0),
        )
    elif optimization_algorithm == "adam":
        return optim.Adam(
            dnn.parameters(),
            lr=params.get("learning_rate", 0.001),
            betas=(params.get("beta1", 0.9), params.get("beta2", 0.999)),
            weight_decay=params.get("weight_decay", 0),
        )
    elif optimization_algorithm == "adadelta":
        return optim.Adadelta(
            dnn.parameters(),
            lr=params.get("learning_rate", 1.0),
            rho=params.get("rho", 0.9),
            weight_decay=params.get("weight_decay", 0),
        )
    else:
        raise ValueError("Unknown optimizer type: %s" % optimization_algorithm)


def precompute_cache(dnn, train_loader, val_loader, config, device="cpu"):
    logger = logging.getLogger(__name__)
    logger.info("Pre-computing and caching outputs.")
    dnn.to(device)
    for i, (idx, t_x, _, _) in enumerate(train_loader):
        t_x = t_x.to(device)
        with torch.no_grad():
            _ = dnn(t_x, cache_ids=idx)
        logger.debug(
            "Pre-computed %08d / %08d training instances.",
            i * train_loader.batch_size + idx.size(0),
            len(train_loader.dataset),
        )
    if not config.get("novalidation", False):
        for i, (idx, t_x, _, _) in enumerate(val_loader):
            t_x = t_x.to(device)
            with torch.no_grad():
                _ = dnn(t_x, cache_ids=idx, validation=True)
            logger.debug(
                "Pre-computed %08d / %08d validation instances",
                i * val_loader.batch_size + idx.size(0),
                len(val_loader.dataset),
            )
    dnn.to(torch.device("cpu"))


def distill(config: DistillationConfiguration) -> None:
    logger = logging.getLogger(__name__)
    device = get_device(config)
    logger.info("Using device: %s", device)
    network = load_network(config.teacher)
    teacher = network.as_pytorch(maintain_weights=True)
    student = transform_network(network, config).as_pytorch()
    is_smaller = _check_smaller(teacher, student)
    if config.get("ensure_reduction", False):
        assert is_smaller, "the student network must be no larger than the teacher"

    train_loader = get_data_loader(config.data.train)
    val_loader = get_data_loader(config.data.validation)

    teacher.eval()
    if config.get("precompute_teacher", False):
        precompute_cache(teacher, train_loader, val_loader, config, device=device)

    iteration = 0
    student_path = config.student.get("path", "/tmp/student.onnx")
    student_path_dir = os.path.dirname(student_path)
    student_path_base = os.path.basename(student_path)
    student_path_base_name, student_path_base_ext = os.path.splitext(student_path_base)
    if not os.path.exists(student_path_dir):
        os.makedirs(student_path_dir)
    if config.get("save_intermediate", False):
        student_path_template = os.path.join(
            student_path_dir,
            "".join((student_path_base_name + ".iter.%d", student_path_base_ext)),
        )
        logger.info(
            "Saving initial student model to %s", (student_path_template % iteration)
        )
        student.export_onnx(student_path_template % iteration)

    student.to(device)
    student.train()

    prediction_type = config.get("type", "classification")
    params = config.parameters
    num_epochs = params.get("epochs", 100)
    logger.debug("Training parameters: %s", params)
    loss = params.get("loss", "KD").lower()
    assert prediction_type == "classification" or loss != "KD"
    loss_fn = get_loss_function(loss, params)
    optimization_algorithm = params.get("optimizer", "SGD").lower()
    optimizer = get_optimizer(optimization_algorithm, student, params)

    best_epoch = 0
    best_val_error = float("inf")
    while iteration < num_epochs:
        iteration += 1
        logger.info(
            "Epoch %d (best epoch = %d, error = %.6f)",
            iteration,
            best_epoch,
            best_val_error,
        )
        train(
            teacher,
            student,
            train_loader,
            loss_fn,
            optimizer,
            device,
            prediction_type,
            config.data.train,
        )
        if config.get("save_intermediate", False):
            student.export_onnx(student_path_template % iteration)
        if not config.get("novalidation", False):
            error = validate(
                teacher,
                student,
                val_loader,
                loss_fn,
                device,
                prediction_type,
                config.data.validation,
            )
            if error["relative"] < best_val_error:
                best_val_error = error["relative"]
                best_epoch = iteration
                student.export_onnx(student_path)
            if error["relative"] < config.get("threshold", float("-inf")):
                break
    if config.get("novalidation", False):
        student.export_onnx(student_path)


def transform_network(network, config):
    logger = logging.getLogger(__name__)
    for i, layer in enumerate(layer for layer in network.layers if not layer.dropped):
        logger.debug("%d: %s", i, layer)
        logger.debug("%d: (shape) %s -> %s", i, layer.input_shape, layer.output_shape)
    for strategy in config.strategies:
        strategy(network)
    for i, layer in enumerate(layer for layer in network.layers if not layer.dropped):
        logger.debug("%d: %s", i, layer)
        logger.debug("%d: (shape) %s -> %s", i, layer.input_shape, layer.output_shape)
    return network


def train(
    teacher, student, data_loader, loss_fn, optimizer, device, prediction_type, config
):
    logger = logging.getLogger(__name__)
    total_loss = 0.0
    example_count = 0.0
    for i, (idx, t_x, s_x, y) in enumerate(data_loader):
        s_x = s_x.to(device)
        y = y.to(device)
        with torch.no_grad():
            teacher_y = teacher(t_x, cache_ids=idx).to(device)
        optimizer.zero_grad()
        student_y = student(s_x)
        loss = loss_fn(teacher_y, student_y, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        example_count += float(idx.size(0))
        if (i + 1) % 25 == 0:
            logger.info(
                "(%7d / %7d): %.6f %.6f",
                (i + 1) * data_loader.batch_size,
                len(data_loader.dataset),
                loss.item() / float(idx.size(0)),
                total_loss / example_count,
            )
        elif (i + 1) % 1 == 0:
            logger.debug(
                "(%7d / %7d): %.6f %.6f",
                (i + 1) * data_loader.batch_size,
                len(data_loader.dataset),
                loss.item() / float(idx.size(0)),
                total_loss / example_count,
            )
        if np.isnan(total_loss):
            logger.error(
                "Loss is `nan`! Outputs: "
                "student=%s (student contains `nan`=%s), "
                "teacher=%s (teacher contains `nan`=%s), target=%s",
                student_y,
                bool(torch.isnan(student_y).any()),
                teacher_y,
                bool(torch.isnan(teacher_y).any()),
                y,
            )
            raise DistillationError("Loss is `nan`!")
        if np.isinf(total_loss):
            logger.error(
                "Loss is infinite! Outputs: "
                "student=%s (student contains `nan`=%s), "
                "teacher=%s (teacher contains `nan`=%s), target=%s",
                student_y,
                bool(torch.isinf(student_y).any()),
                teacher_y,
                bool(torch.isinf(teacher_y).any()),
                y,
            )
            raise DistillationError("Loss is infinite!")
    logger.info("training loss: %.6f", total_loss / example_count)


def validate(teacher, student, data_loader, loss_fn, device, prediction_type, config):
    logger = logging.getLogger(__name__)
    student_error = 0.0
    teacher_error = 0.0
    relative_error = 0.0
    num_samples = 0.0
    relative_num_samples = 0.0
    with torch.no_grad():
        for i, (idx, t_x, s_x, target) in enumerate(data_loader):
            s_x = s_x.to(device)
            target = target.to(device)
            teacher_y = teacher(t_x, cache_ids=idx, validation=True).to(device)
            student_y = student(s_x)
            num_samples += target.size(0)
            if prediction_type == "classification":
                teacher_pred = teacher_y.argmax(-1)
                student_pred = student_y.argmax(-1)
                teacher_correct = teacher_pred == target
                student_correct = student_pred == target
                count = target.size(0)
                relative_count = teacher_correct.sum(dtype=torch.float)
                relative_num_samples += relative_count

                student_error += count - student_correct.sum(dtype=torch.float)
                teacher_error += count - teacher_correct.sum(dtype=torch.float)
                relative_error += relative_count - (
                    teacher_correct & student_correct
                ).sum(dtype=torch.float)

                error = {
                    "student": (student_error / num_samples).item(),
                    "teacher": (teacher_error / num_samples).item(),
                    "relative": (relative_error / relative_num_samples).item()
                    if relative_num_samples > 0
                    else float("nan"),
                }
            else:
                student_error += loss_fn(student_y, target.reshape(student_y.shape))
                teacher_error += loss_fn(teacher_y, target.reshape(teacher_y.shape))
                relative_error += loss_fn(student_y, teacher_y)
                error = {
                    "student": (student_error / num_samples).item(),
                    "teacher": (teacher_error / num_samples).item(),
                    "relative": (relative_error / num_samples).item(),
                }
            if (i + 1) % 1 == 0:
                logger.debug(
                    "(%7d / %7d): student=%.6f, teacher=%.6f, relative=%.6f",
                    (i + 1) * data_loader.batch_size,
                    len(data_loader.dataset),
                    error["student"],
                    error["teacher"],
                    error["relative"],
                )
    logger.info(
        "validation error: student=%.6f, teacher=%.6f, relative=%.6f",
        error["student"],
        error["teacher"],
        error["relative"],
    )
    return error
