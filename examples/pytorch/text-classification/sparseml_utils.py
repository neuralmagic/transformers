from typing import Any

import numpy
import torch
import torch.nn.functional as F

from sparseml.pytorch.utils import ModuleExporter

from transformers.modeling_outputs import SequenceClassifierOutput

class GLUEModuleExporter(ModuleExporter):
    """
    Module exporter class for Sequence Classification
    """

    @classmethod
    def get_output_names(self, out: Any):
        if not isinstance(out, SequenceClassifierOutput):
            raise ValueError("Expected SequenceClassifierOutput, got {type(out)}")
        expected = ["logits"]
        if numpy.any([name for name in expected if name not in out]):
            raise ValueError("Expected output names not found in model output")
        return expected
