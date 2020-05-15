import toml

from functools import partial

from .. import logging
from ..config import parse, Configuration
from ..distillation import strategies as distillation_strategies


class DistillationConfiguration(Configuration):
    @property
    def strategies(self):
        for strat, strat_config in self.config.get("strategies", {}).items():
            if isinstance(strat_config, list):
                for s_config in strat_config:
                    yield partial(getattr(distillation_strategies, strat), **s_config)
            else:
                yield partial(getattr(distillation_strategies, strat), **strat_config)

    @property
    def student(self):
        student_config = self.config.get("student", {})
        return Configuration(student_config)

    @property
    def teacher(self):
        teacher_config = self.config.get("teacher", None)
        if teacher_config is not None:
            return Configuration(teacher_config)
        raise ValueError("No teacher configuration defined")

    @property
    def data(self):
        data_config = self.config.get("data", None)
        if data_config is not None:
            return DataConfiguration(data_config)
        raise ValueError("No data configuration defined")

    @property
    def parameters(self):
        parameters = self.config.get("parameters", {})
        return parameters


class DataConfiguration(Configuration):
    @property
    def batch_size(self):
        return self.batchsize

    @property
    def batchsize(self):
        batch_size = self.config.get("batchsize", None)
        if batch_size is not None:
            return batch_size
        raise ValueError("No batch size defined")

    @property
    def path(self):
        path = self.config.get("path", None)
        if path is not None:
            return path
        raise ValueError("No data path defined")

    @property
    def student(self):
        if self.config.get("_STAGE", None) is None:
            raise ValueError("No student config for unknown data stage.")
        if self.config.get("_NN", None) is not None:
            raise ValueError(
                "Already in data configuration for %s network." % self.config.get("_NN")
            )
        data_config = self.config.copy()
        student_config = self.config.get("student", {})
        data_config.update(student_config)
        data_config["_NN"] = "student"
        return DataConfiguration(data_config)

    @property
    def teacher(self):
        if self.config.get("_STAGE", None) is None:
            raise ValueError("No teacher config for unknown data stage.")
        if self.config.get("_NN", None) is not None:
            raise ValueError(
                "Already in data configuration for %s network." % self.config.get("_NN")
            )
        data_config = self.config.copy()
        teacher_config = self.config.get("teacher", {})
        data_config.update(teacher_config)
        data_config["_NN"] = "teacher"
        return DataConfiguration(data_config)

    @property
    def test(self):
        test_config = self.config.get("test", None)
        if test_config is not None:
            data_config = self.config.copy()
            data_config.update(test_config)
            data_config["_STAGE"] = "test"
            return DataConfiguration(data_config)
        raise ValueError("No test configuration defined")

    @property
    def train(self):
        train_config = self.config.get("train", None)
        if train_config is not None:
            data_config = self.config.copy()
            data_config.update(train_config)
            data_config["_STAGE"] = "train"
            return DataConfiguration(data_config)
        raise ValueError("No train configuration defined")

    @property
    def transform(self):
        transform_config = self.config.get("transform", {})
        if "presized" not in transform_config:
            transform_config["presized"] = self.config.get("presized", True)
        if "student" not in transform_config:
            transform_config["student"] = {}
        for key, value in transform_config.items():
            if (
                key not in ["student", "teacher"]
                and key not in transform_config["student"]
            ):
                transform_config["student"][key] = value
        if "teacher" not in transform_config:
            transform_config["teacher"] = {}
        for key, value in transform_config.items():
            if (
                key not in ["student", "teacher"]
                and key not in transform_config["teacher"]
            ):
                transform_config["teacher"][key] = value
        return transform_config

    @property
    def validation(self):
        validation_config = self.config.get("validation", None)
        if validation_config is not None:
            data_config = self.config.copy()
            data_config.update(validation_config)
            data_config["_STAGE"] = "validation"
            return DataConfiguration(data_config)
        raise ValueError("No validation configuration defined")

