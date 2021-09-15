from typing import Any

import numpy
import torch
import torch.nn.functional as F

from sparseml.pytorch.utils import ModuleExporter
from trainer_qa import QuestionAnsweringTrainer
from transformers.modeling_outputs import QuestionAnsweringModelOutput
from transformers.sparse import SparseMLTrainer


class SparseMLQATrainer(SparseMLTrainer, QuestionAnsweringTrainer):
    """
    Question Answering trainer with SparseML integration

    :param recipe: recipe for model sparsification
    :param teacher: teacher model for distillation
    :param distill_hardness: ratio of loss by teacher targets (between 0 and 1)
    :param distill_temperature: temperature for distillation
    :param args, kwargs: arguments passed into parent class
    """
<<<<<<< HEAD

    def __init__(
        self, model_name_or_path, recipes, teacher=None, distill_hardness=0.5, distill_temperature=2.0, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.model_name_or_path = str(model_name_or_path)
        self.recipes = [recipe for recipe in recipes if recipe]
        self.teachers = teacher
        self.multi_gpu = False
        if torch.cuda.device_count() and teacher != None:
            self.multi_gpu = True
            self.num_gpus = torch.cuda.device_count()
            self.teachers = [teacher for i in range(self.num_gpus)]
            for i in range(self.num_gpus):
                self.teachers[i] = self.teachers[i].to(i)
        self.distill_hardness = distill_hardness
        self.distill_temperature = distill_temperature
        self.criterion = torch.nn.CrossEntropyLoss()

        manager = None
        modifiers = []
        for recipe in self.recipes:
            manager = ScheduledModifierManager.from_yaml(recipe, modifiers)
            modifiers = manager.modifiers
        self.manager = manager

        self.loggers = None
        if self.recipes is not None:
            loggers = []
            if "wandb" in self.args.report_to:
                loggers.append(logger.WANDBLogger())
            self.loggers = loggers

    def apply_recipes(self, epoch=0.0):
        """
        Apply recipes and sparsification related parameters to the model
        """
        if self.manager is not None:
            org_state_dict = self.model.state_dict()
            self.manager.initialize(self.model, epoch=epoch, loggers=self.loggers)
            new_state_dict = self.model.state_dict()
            new_params = [p for p in new_state_dict.keys() if p not in org_state_dict]

            if os.path.isdir(self.model_name_or_path):
                if os.path.isfile(os.path.join(self.model_name_or_path, WEIGHTS_NAME)):
                    archive_file = os.path.join(self.model_name_or_path, WEIGHTS_NAME)
                    state_dict = torch.load(archive_file, map_location="cpu")
                    new_params_to_init = [p for p in new_params if p in state_dict.keys()]
                    if new_params_to_init:
                        # If we're here, the assumption is that all the new parameters introduced
                        # by the recipes are available to be restore from the checkpoint---this is
                        # case of evaluating pruned or pruned quantized models
                        # Otherwise, we're in use cases such as quantizing a block pruned model in which
                        # new parameters need to be initialized and trained during the QAT process
                        _, missing_keys, unexpected_keys, _ = BertForQuestionAnswering._load_state_dict_into_model(
                            self.model, state_dict, self.model_name_or_path, _fast_init=False
                        )
                        if missing_keys or unexpected_keys:
                            raise RuntimeError(
                                "Unexpected or missing keys detected when applying recipes to models\n"
                                f"Missing keys: {missing_keys}\n"
                                f"Unexpected keys: {unexpected_keys}\n"
                            )

    def create_optimizer(self):
        """
        Create optimizer customized using SparseML
        """
        super().create_optimizer()
        if not self.recipes:
            return
        steps_per_epoch = math.ceil(
            len(self.train_dataset) / (self.args.per_device_train_batch_size * self.args._n_gpu)
        )
        self.args.num_train_epochs = float(self.manager.max_epochs)
        if hasattr(self, "scaler"):
            self.scaler = self.manager.modify(
                self.model, self.optimizer, steps_per_epoch=steps_per_epoch, wrap_optim=self.scaler
            )
        else:
            self.optimizer = ScheduledOptimizer(
                self.optimizer, self.model, self.manager, steps_per_epoch=steps_per_epoch, loggers=self.loggers
            )

=======
    
>>>>>>> 6f798792243fbfe569f8605709b275ffd0198dfd
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Computing loss using teacher/student distillation
        """
        if not self.recipes and self.teachers is None:
            return super().compute_loss(model, inputs, return_outputs=return_outputs)

        outputs = model(**inputs)
        if self.teachers is None:
            loss = outputs["loss"]
        else:
            start_logits_student = outputs["start_logits"]
            end_logits_student = outputs["end_logits"]
            start_logits_label = inputs["start_positions"]
            end_logits_label = inputs["end_positions"]
            if self.multi_gpu:
                input_ids = torch.split(inputs['input_ids'], int(inputs['input_ids'].shape[0]/self.num_gpus))
                start_logits_teacher = torch.empty((0,inputs['input_ids'].shape[1]), dtype=torch.int32, device='cuda')
                end_logits_teacher = torch.empty((0,inputs['input_ids'].shape[1]), dtype=torch.int32, device='cuda')
                for i in range(self.num_gpus):
                    with torch.no_grad():
                        input_device = self.teachers[i].device
                        teacher_output = self.teachers[i](input_ids[i].to(input_device))
                    start_logits_teacher = torch.cat((start_logits_teacher, teacher_output["start_logits"].to('cuda')), dim=0)
                    end_logits_teacher = torch.cat((end_logits_teacher, teacher_output["end_logits"].to('cuda')), dim=0)
            else: # CPU or single GPU
                input_device = inputs["input_ids"].device
                self.teachers = self.teachers.to(input_device)
                with torch.no_grad():
                    teacher_output = self.teachers(
                        input_ids=inputs["input_ids"],
                        token_type_ids=inputs["token_type_ids"],
                        attention_mask=inputs["attention_mask"],
                    )
                start_logits_teacher = teacher_output["start_logits"]
                end_logits_teacher = teacher_output["end_logits"]

            loss_start = (
                F.kl_div(
                    input=F.log_softmax(start_logits_student / self.distill_temperature, dim=-1),
                    target=F.softmax(start_logits_teacher / self.distill_temperature, dim=-1),
                    reduction="batchmean",
                )
                * (self.distill_temperature ** 2)
            )
            loss_end = (
                F.kl_div(
                    input=F.log_softmax(end_logits_student / self.distill_temperature, dim=-1),
                    target=F.softmax(end_logits_teacher / self.distill_temperature, dim=-1),
                    reduction="batchmean",
                )
                * (self.distill_temperature ** 2)
            )
            teacher_loss = (loss_start + loss_end) / 2.0
            loss_start = self.criterion(start_logits_student, start_logits_label)
            loss_end = self.criterion(end_logits_student, end_logits_label)
            label_loss = (loss_start + loss_end) / 2.0
            loss = ((1 - self.distill_hardness) * label_loss) + (self.distill_hardness * teacher_loss)
        return (loss, outputs) if return_outputs else loss


class QuestionAnsweringModuleExporter(ModuleExporter):
    """
    Module exporter class for Question Answering
    """

    @classmethod
    def get_output_names(self, out: Any):
        if not isinstance(out, QuestionAnsweringModelOutput):
            raise ValueError("Expected QuestionAnsweringModelOutput, got {type(out)}")
        expected = ["start_logits", "end_logits"]
        if numpy.any([name for name in expected if name not in out]):
            raise ValueError("Expected output names not found in model output")
        return expected
