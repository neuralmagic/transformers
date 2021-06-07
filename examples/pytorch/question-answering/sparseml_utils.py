import collections
import math
import torch
import torch.nn.functional as F
import numpy
from trainer_qa import QuestionAnsweringTrainer

from sparseml.pytorch.optim.manager import ScheduledModifierManager
from sparseml.pytorch.optim.optimizer import ScheduledOptimizer

from sparseml.pytorch.utils import logger


class SparseMLQATrainer(QuestionAnsweringTrainer):
    """
    Question Answering trainer with customized optimizer using SparseML

    :param nm_prune_config: recipe for model sparsification
    :param args, kwargs: arguments passed into parent class
    """

    def __init__(self, nm_prune_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nm_prune_config = nm_prune_config
        self.manager = None
        loggers = []
        if "wandb" in self.args.report_to:
            loggers.append(logger.WANDBLogger())
        self.loggers = loggers

    def create_optimizer(self):
        """
        Create optimizer customized using SparseML
        """
        super().create_optimizer()
        steps_per_epoch = math.ceil(
            len(self.train_dataset) / (self.args.per_device_train_batch_size * self.args._n_gpu)
        )
        self.manager = ScheduledModifierManager.from_yaml(self.nm_prune_config)
        self.args.num_train_epochs = float(self.manager.max_epochs)
        if hasattr(self, "scaler"):
            self.manager.initialize(self.model, epoch=0.0, loggers=self.loggers)
            self.scaler = self.manager.modify(
                self.model, self.optimizer, steps_per_epoch=steps_per_epoch, wrap_optim=self.scaler
            )
        else:
            self.optimizer = ScheduledOptimizer(
                self.optimizer, self.model, self.manager, steps_per_epoch=steps_per_epoch, loggers=self.loggers
            )


class SparseMLDistillQATrainer(SparseMLQATrainer):
    """
    Question Answering trainer using distilation with customized optimizer using SparseML

    :param nm_prune_config: recipe for model sparsification
    :param teacher: teacher model
    :param distill_hardness: weight of the teacher loss
    :param temperature: temperature used for loss
    :param args, kwargs: arguments passed into parent class
    """

    def __init__(self, nm_prune_config, teacher=None, distill_hardness=0.5, temperature=2.0, *args, **kwargs):
        super().__init__(nm_prune_config, *args, **kwargs)
        self.teacher = teacher
        self.distill_hardness = distill_hardness
        self.temperature = temperature
        self.criterion = torch.nn.CrossEntropyLoss()

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Computing loss using teacher/student distillation
        """
        outputs = model(**inputs)
        loss = outputs["loss"]
        if self.teacher is not None:
            input_device = inputs["input_ids"].device
            self.teacher = self.teacher.to(input_device)
            start_logits_student = outputs["start_logits"]
            end_logits_student = outputs["end_logits"]
            start_logits_label = inputs["start_positions"]
            end_logits_label = inputs["start_positions"]
            with torch.no_grad():
                teacher_output = self.teacher(
                    input_ids=inputs["input_ids"],
                    token_type_ids=inputs["token_type_ids"],
                    attention_mask=inputs["attention_mask"],
                )
            start_logits_teacher = teacher_output["start_logits"]
            end_logits_teacher = teacher_output["end_logits"]
            loss_start = (
                F.kl_div(
                    input=F.log_softmax(start_logits_student / self.temperature, dim=-1),
                    target=F.softmax(start_logits_teacher / self.temperature, dim=-1),
                    reduction="batchmean",
                )
                * (self.temperature ** 2)
            )
            loss_end = (
                F.kl_div(
                    input=F.log_softmax(end_logits_student / self.temperature, dim=-1),
                    target=F.softmax(end_logits_teacher / self.temperature, dim=-1),
                    reduction="batchmean",
                )
                * (self.temperature ** 2)
            )
            teacher_loss = (loss_start + loss_end) / 2.0
            loss_start = self.criterion(start_logits_student, start_logits_label)
            loss_end = self.criterion(end_logits_student, end_logits_label)
            label_loss = (loss_start + loss_end) / 2.0
            loss = ((1 - self.distill_hardness) * label_loss) + (self.distill_hardness * teacher_loss)
        return (loss, outputs) if return_outputs else loss


def convert_example_to_features(example, tokenizer, max_seq_length, doc_stride, max_query_length=30):
    """
    Convert example to features, used for onnx export
    """
    Feature = collections.namedtuple(
        "Feature",
        [
            "unique_id",
            "tokens",
            "example_index",
            "token_to_orig_map",
            "token_is_max_context",
        ],
    )
    extra = []
    unique_id = 0
    query_tokens = tokenizer.tokenize(example["question"])[0:max_query_length]
    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example["context"]):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)
    max_tokens_for_doc = max_seq_length - len(query_tokens) - 3
    _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
        length = len(all_doc_tokens) - start_offset
        if length > max_tokens_for_doc:
            length = max_tokens_for_doc
        doc_spans.append(_DocSpan(start=start_offset, length=length))
        if start_offset + length == len(all_doc_tokens):
            break
        start_offset += min(length, doc_stride)
    for (doc_span_index, doc_span) in enumerate(doc_spans):
        tokens = []
        token_to_orig_map = {}
        token_is_max_context = {}
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in query_tokens:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)
        for i in range(doc_span.length):
            split_token_index = doc_span.start + i
            token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]
            is_max_context = _check_is_max_context(doc_spans, doc_span_index, split_token_index)
            token_is_max_context[len(tokens)] = is_max_context
            tokens.append(all_doc_tokens[split_token_index])
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
        feature = Feature(
            unique_id=unique_id,
            tokens=tokens,
            example_index=0,
            token_to_orig_map=token_to_orig_map,
            token_is_max_context=token_is_max_context,
        )
        extra.append(feature)
        unique_id += 1
        # extra is used as additional data but sparseml doesn't support it
    return (
        torch.from_numpy(numpy.array([numpy.array(input_ids, dtype=numpy.int64)])),
        torch.from_numpy(numpy.array([numpy.array(input_mask, dtype=numpy.int64)])),
        torch.from_numpy(numpy.array([numpy.array(segment_ids, dtype=numpy.int64)])),
    )


def _check_is_max_context(doc_spans, cur_span_index, position):
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index
    return cur_span_index == best_span_index
