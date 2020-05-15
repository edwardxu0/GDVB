import argparse
import unittest

import r4v.distillation as r4v
import r4v.logging as logging
from tests.utils import measure_performance

from r4v.config import parse

logging.initialize("r4v", argparse.Namespace(quiet=True, debug=False, verbose=False))


class TestDistillConvBatchNormNetwork(unittest.TestCase):
    def distill_and_check_accuracy(self, config_path):
        config = parse(config_path)
        distillation_config = config.distillation
        distillation_config.config["novalidation"] = True
        with logging.suppress():
            r4v.distill(distillation_config)
        student_accuracy = measure_performance(distillation_config, student=True)[
            "accuracy"
        ]
        teacher_accuracy = measure_performance(distillation_config, teacher=True)[
            "accuracy"
        ]
        self.assertAlmostEqual(teacher_accuracy, 0.99, 2)
        self.assertLessEqual(round(abs(teacher_accuracy - student_accuracy), 2), 0.05)

    def test_to_same(self):
        self.distill_and_check_accuracy(
            "tests/test_distill_conv_bn/conv_bn_to_same.toml"
        )

    def test_conv_bn_to_fc(self):
        self.distill_and_check_accuracy("tests/test_distill_conv_bn/conv_bn_to_fc.toml")

    def test_conv_bn_to_conv_bn(self):
        self.distill_and_check_accuracy(
            "tests/test_distill_conv_bn/conv_bn_to_conv_bn.toml"
        )


if __name__ == "__main__":
    unittest.main(buffer=True)
