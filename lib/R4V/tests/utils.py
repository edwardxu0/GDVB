import onnx
import toml
import torch

from r4v.distillation.data import get_data_loader
from r4v.nn import load_network


def measure_performance(config, *, student=None, teacher=None):
    data_loader = get_data_loader(config.data.validation)

    if student:
        model = load_network(
            {
                "model": config.student["path"],
                "input_shape": config.teacher["input_shape"],
            }
        ).as_pytorch(maintain_weights=True)
    elif teacher:
        model = load_network(config.teacher).as_pytorch(maintain_weights=True)
    else:
        raise ValueError(
            "must specify whether to measure student of teacher performance"
        )

    device = torch.device("cpu")
    if config.get("cuda", False) and torch.cuda.is_available():
        device = torch.device("cuda")
    model.to(device)

    model.eval()

    prediction_type = config.get("type", "classification")
    num_correct = 0.0
    num_samples = 0.0
    loss = 0.0
    for i, (idx, _, sx, target) in enumerate(data_loader):
        y = model(sx.to(device)).squeeze()
        target = target.to(device)
        num_samples += target.shape[0]
        if prediction_type == "classification":
            pred = y.argmax(-1)
            correct = pred == target
            num_correct += correct.sum(dtype=torch.float)
            performance = {"accuracy": (num_correct / num_samples).item()}
        else:
            loss += ((y - target) ** 2).sum()
            performance = {"loss": (loss / num_samples).item()}
    return performance
