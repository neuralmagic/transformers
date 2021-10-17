from typing import Any

import numpy

from sparseml.pytorch.utils import ModuleExporter
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.sparse import SparseMLTrainer


class SparseMLTokenClassificationTrainer(SparseMLTrainer):
    """
    Token Classification trainer with SparseML integration

    :param recipe: recipe for model sparsification
    :param teacher: teacher model for distillation
    :param distill_hardness: ratio of loss by teacher targets (between 0 and 1)
    :param distill_temperature: temperature for distillation
    :param args, kwargs: arguments passed into parent class
    """

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Computing loss using teacher/student distillation
        """
        if not self.recipes or self.teacher is None:
            return super().compute_loss(model, inputs, return_outputs=return_outputs)
        student_outputs = model(**inputs)
        loss = student_outputs["loss"]

        steps_in_epoch = -1  # Unused
        loss = self.manager.loss_update(
            loss,
            model,
            self.optimizer,
            self.state.epoch,
            steps_in_epoch,
            global_step=self.state.global_step,
            student_outputs=student_outputs,
            teacher_inputs=inputs,
        )
        return (loss, student_outputs) if return_outputs else loss


class TokenClassificationModuleExporter(ModuleExporter):
    """
    Module exporter class for Question Answering
    """

    @classmethod
    def get_output_names(self, out: Any):
        if not isinstance(out, TokenClassifierOutput):
            raise ValueError(f"Expected TokenClassifierOutput, got {type(out)}")
        expected = ["logits"]
        if numpy.any([name for name in expected if name not in out]):
            raise ValueError("Expected output names not found in model output")
        return expected
