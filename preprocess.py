#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence speech recognition
with 🤗 Datasets' streaming mode.
"""
# You can also adapt this script for your own sequence to sequence speech
# recognition task. Pointers for this are left as comments.

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import datasets
import torch
from datasets import Dataset, concatenate_datasets, DatasetDict, IterableDatasetDict, interleave_datasets, load_dataset, load_from_disk, concatenate_datasets
from torch.utils.data import IterableDataset

import evaluate
import transformers
from transformers import (
    AutoConfig,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    set_seed,
)

from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers.trainer_pt_utils import IterableDatasetShard
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import gcsfs

from google.cloud import storage
from google.oauth2 import service_account

import os


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.25.0.dev0")

require_version("datasets>=1.18.2", "To fix: pip install -r examples/pytorch/speech-recognition/requirements.txt")

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    feature_extractor_name: Optional[str] = field(
        default=None, metadata={"help": "feature extractor name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    freeze_feature_encoder: bool = field(
        default=True, metadata={"help": "Whether to freeze the feature encoder layers of the model."}
    )
    freeze_encoder: bool = field(
        default=False, metadata={"help": "Whether to freeze the entire encoder of the seq2seq model."}
    )
    forced_decoder_ids: List[List[int]] = field(
        default=None,
        metadata={
            "help": (
                "A list of pairs of integers which indicates a mapping from generation indices to token indices "
                "that will be forced before sampling. For example, [[0, 123]] means the first generated token "
                "will always be a token of index 123."
            )
        },
    )
    suppress_tokens: List[int] = field(
        default=None, metadata={"help": "A list of tokens that will be suppressed at generation."}
    )
    model_index_name: str = field(default=None, metadata={"help": "Pretty name for the model card."})
    # resume_from_checkpoint: bool = field(
    #     default=False,
    #     metadata={'help': 'Use to continue train in checkpoint when run else train from scratch'}
    # )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: List[str] = field(
        default=None, metadata={"help": "The list of names of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    audio_column_name: str = field(
        default="audio",
        metadata={"help": "The name of the dataset column containing the audio data. Defaults to 'audio'"},
    )
    text_column_name: str = field(
        default="text",
        metadata={"help": "The name of the dataset column containing the text data. Defaults to 'text'"},
    )
    max_duration_in_seconds: float = field(
        default=20.0,
        metadata={
            "help": (
                "Truncate audio files that are longer than `max_duration_in_seconds` seconds to"
                " 'max_duration_in_seconds`"
            )
        },
    )
    min_duration_in_seconds: float = field(
        default=0.0, metadata={"help": "Filter audio files that are shorter than `min_duration_in_seconds` seconds"}
    )
    train_split_name: str = field(
        default="train",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    eval_split_name: str = field(
        default="validation",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    do_lower_case: bool = field(
        default=False,
        metadata={"help": "Whether the target text should be lower cased."},
    )
    do_remove_punctuation: bool = field(
        default=False,
        metadata={"help": "Whether the target text should be striped of punctuation."},
    )
    do_normalize_eval: bool = field(
        default=True,
        metadata={"help": "Whether to normalise the references and predictions in the eval WER calculation."},
    )
    language: str = field(
        default=None,
        metadata={
            "help": (
                "Language for multilingual fine-tuning. This argument should be set for multilingual fine-tuning "
                "only. For English speech recognition, it should be set to `None`."
            )
        },
    )
    task: str = field(
        default="transcribe",
        metadata={"help": "Task, either `transcribe` for speech recognition or `translate` for speech translation."},
    )
    shuffle_buffer_size: Optional[int] = field(
        default=500,
        metadata={
            "help": (
                "The number of streamed examples to download before shuffling them. The large the buffer, "
                "the closer it is to real offline shuffling."
            )
        },
    )
    streaming: bool = field(
        default=True,
        metadata={"help": "Whether to use streaming mode to load and pre-process the data."},
    )
    cache_data_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the datasets downloaded from huggingface.co"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    output_data_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the preprocessed dataset to disk."},
    )
    input_preprocessed_data_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to load the preprocessed dataset to disk."},
    )
    augmenting: bool = field(
        default=False,
        metadata={"help": "Whether to use data augmentation."},
    )
    spliting_custome_dataset: bool = field(
        default=None,
        metadata={"help": "Enable when the data do not split train and test"},
    )
    test_size_percent: float = field(
        default=None,
        metadata={"help": "The test size percent to split data into train and test"},
    )



@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor ([`WhisperProcessor`])
            The processor used for processing the data.
        decoder_start_token_id (`int`)
            The begin-of-sentence of the decoder.
    """

    processor: Any
    decoder_start_token_id: int
    data: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        model_input_name = self.processor.model_input_names[0]
        input_features = [{model_input_name: feature[model_input_name]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]

        max_label_len_train = max([len(label) for label in self.data['train']['labels']])
        max_label_len_test = max([len(label) for label in self.data['eval']['labels']])
        max_label_len = max(max_label_len_train, max_label_len_test)

        labels_batch = self.processor.tokenizer.pad(label_features, padding='max_length', max_length=max_label_len, return_tensors="pt")

        # labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def load_maybe_streaming_dataset(dataset_name, dataset_config_name, split="train", streaming=True, cache_dir = None, **kwargs):
    """
    Utility function to load a dataset in streaming mode. For datasets with multiple splits,
    each split is loaded individually and then splits combined by taking alternating examples from
    each (interleaving).
    """
    if "+" in split:
        # load multiple splits separated by the `+` symbol with streaming mode
        dataset_splits = [
            load_dataset(dataset_name, dataset_config_name, split=split_name, streaming=streaming, cache_dir=cache_dir, **kwargs)
            for split_name in split.split("+")
        ]
        # interleave multiple splits to form one dataset
        interleaved_dataset = interleave_datasets(dataset_splits)
        return interleaved_dataset
    else:
        # load a single split *with* streaming mode
        dataset = load_dataset(dataset_name, dataset_config_name, split=split, streaming=streaming, cache_dir=cache_dir, **kwargs)
        return dataset


def main():
    # 1. Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_seq2seq", model_args, data_args)

    # 2. Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # 3. Detecting last checkpoint and eventually continue from last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # 4. Load dataset
    raw_datasets = IterableDatasetDict() if data_args.streaming else DatasetDict()
    print(data_args.dataset_name)
    dataset_train = {
        "audio": [],
        "transcription": [],
    }

    dataset_val = {
        "audio": [],
        "transcription": [],
    }
    final_dataset_train = Dataset.from_dict(dataset_train)
    final_dataset_val = Dataset.from_dict(dataset_val)

    
    for ds_name in data_args.dataset_name:
        if (data_args.input_preprocessed_data_dir is None):
            if training_args.do_train:
                raw_datasets["train"] = load_maybe_streaming_dataset(
                    ds_name,
                    data_args.dataset_config_name,
                    split=data_args.train_split_name,
                    use_auth_token=True if model_args.use_auth_token else None,
                    streaming=data_args.streaming,
                    cache_dir=data_args.cache_data_dir,
                )

            if training_args.do_eval:
                raw_datasets["validation"] = load_maybe_streaming_dataset(
                    ds_name,
                    data_args.dataset_config_name,
                    split=data_args.eval_split_name,
                    use_auth_token=True if model_args.use_auth_token else None,
                    streaming=data_args.streaming,
                    cache_dir=data_args.cache_data_dir,
                )

            if data_args.spliting_custome_dataset:
                raw_datasets = load_dataset(ds_name, num_proc=4).train_test_split(test_size=data_args.test_size_percent)
                raw_datasets.rename_column('text', 'eval')

            raw_datasets_features = list(next(iter(raw_datasets.values())).features.keys())

            if data_args.audio_column_name not in raw_datasets_features:
                raise ValueError(
                    f"--audio_column_name '{data_args.audio_column_name}' not found in dataset '{ds_name}'. "
                    "Make sure to set `--audio_column_name` to the correct audio column - one of "
                    f"{', '.join(raw_datasets_features)}."
                )

            if data_args.text_column_name not in raw_datasets_features:
                raise ValueError(
                    f"--text_column_name {data_args.text_column_name} not found in dataset '{ds_name}'. "
                    "Make sure to set `--text_column_name` to the correct text column - one of "
                    f"{', '.join(raw_datasets_features)}."
                )

            final_dataset_train = concatenate_datasets([final_dataset_train, raw_datasets["train"]])
            final_dataset_val = concatenate_datasets([final_dataset_val, raw_datasets["validation"]])

    raw_datasets = DatasetDict({"train": final_dataset_train,
                                "validation": final_dataset_val})
    print(raw_datasets)
            
    # 5. Load pretrained model, tokenizer, and feature extractor, single speech processor
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    config.update({"forced_decoder_ids": model_args.forced_decoder_ids, "suppress_tokens": model_args.suppress_tokens})

    if training_args.gradient_checkpointing:
        config.update({"use_cache": False})

    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        model_args.feature_extractor_name if model_args.feature_extractor_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = WhisperTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if model_args.freeze_feature_encoder:
        model.freeze_feature_encoder()

    if model_args.freeze_encoder:
        model.freeze_encoder()

    if data_args.language is not None:
        # We only need to set the task id when the language is specified (i.e. in a multilingual setting)
        tokenizer.set_prefix_tokens(language=data_args.language, task=data_args.task)

    if is_main_process(training_args.local_rank):
        # save feature extractor, tokenizer and config
        feature_extractor.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
        config.save_pretrained(training_args.output_dir)

    processor = WhisperProcessor.from_pretrained(training_args.output_dir)

    # 6. Resample speech dataset if necessary
    if (data_args.input_preprocessed_data_dir is None):
        dataset_sampling_rate = next(iter(raw_datasets.values())).features[data_args.audio_column_name].sampling_rate
        if dataset_sampling_rate != feature_extractor.sampling_rate:
            raw_datasets = raw_datasets.cast_column(
                data_args.audio_column_name, datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate)
            )

        # 7. Preprocessing the datasets.
        # We need to read the audio files as arrays and tokenize the targets.
        max_input_length = data_args.max_duration_in_seconds * feature_extractor.sampling_rate
        min_input_length = data_args.min_duration_in_seconds * feature_extractor.sampling_rate
        audio_column_name = data_args.audio_column_name
        text_column_name = data_args.text_column_name
        model_input_name = feature_extractor.model_input_names[0]
        do_lower_case = data_args.do_lower_case
        do_remove_punctuation = data_args.do_remove_punctuation
        normalizer = BasicTextNormalizer()  # 'official' text normalizer from OpenAI

        if data_args.max_train_samples is not None:
            raw_datasets["train"] = (
                raw_datasets["train"].take(data_args.max_train_samples)
                if data_args.streaming
                else raw_datasets["train"].select(range(data_args.max_train_samples))
            )

        if data_args.max_eval_samples is not None:
            raw_datasets["eval"] = (
                raw_datasets["eval"].take(data_args.max_eval_samples)
                if data_args.streaming
                else raw_datasets["eval"].select(range(data_args.max_eval_samples))
            )


        augment_waveform = Compose([
            AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.015, p=0.3),
            TimeStretch(min_rate=0.8, max_rate=1.25, p=0.3, leave_length_unchanged=False),
            PitchShift(min_semitones=-4, max_semitones=4, p=0.3),
            ])

        def augment_dataset(batch):

            audio = batch["audio"]["array"]
            # apply augmentation
            augmented_audio = augment_waveform(samples=audio, sample_rate=16000)

            batch["audio"]["array"] = augmented_audio

            return batch


        def prepare_dataset(batch):
            # process audio
            audio = batch[audio_column_name]
            inputs = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"])
            # process audio length
            batch[model_input_name] = inputs.get(model_input_name)[0]
            batch["input_length"] = len(audio["array"])

            # process targets
            input_str = batch[text_column_name].lower() if do_lower_case else batch[text_column_name]
            if do_remove_punctuation:
                input_str = normalizer(input_str).strip()
            batch["labels"] = tokenizer(input_str).input_ids
            return batch

        with training_args.main_process_first(desc="dataset map pre-processing"):
            if (data_args.augmenting):
                raw_augmented_datasets = raw_datasets.map(augment_dataset, num_proc=data_args.preprocessing_num_workers)
                raw_datasets["train"] = concatenate_datasets([raw_augmented_datasets["train"],raw_augmented_datasets["eval"]])
            raw_datasets_features.remove(audio_column_name)
            vectorized_datasets = raw_datasets.map(
                prepare_dataset,
                remove_columns=raw_datasets_features,
                num_proc=data_args.preprocessing_num_workers,
            ).with_format("torch")

            if training_args.do_train and data_args.streaming:
                # manually shuffle if streaming (done by the trainer for non-streaming)
                vectorized_datasets["train"] = vectorized_datasets["train"].shuffle(
                    buffer_size=data_args.shuffle_buffer_size,
                    seed=training_args.seed,
                )

        # filter training data that is shorter than min_input_length or longer than
        # max_input_length
        def is_audio_in_length_range(length):
            return min_input_length < length < max_input_length


        vectorized_datasets['train'] = vectorized_datasets['train'].filter(
            is_audio_in_length_range,
            input_columns=["input_length"],
        )
    else:
        vectorized_datasets = load_from_disk(data_args.input_preprocessed_data_dir)

    max_label_length = 448

    def filter_labels(labels):
        """Filter label sequences longer than max length"""
        return len(labels) < max_label_length

    vectorized_datasets = vectorized_datasets.filter(filter_labels, input_columns=["labels"])

    # 8. Load Metric
    wer_metric = evaluate.load("wer")
    # cer_metric = evaluate.load("cer")
    do_normalize_eval = data_args.do_normalize_eval

    def compute_metrics(pred):
        pred_ids = pred.predictions

        pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        # we do not want to group tokens when computing the metrics
        label_str = tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)

        if do_normalize_eval:
            pred_str = [normalizer(pred) for pred in pred_str]
            label_str = [normalizer(label) for label in label_str]

        wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
        # cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}
    
    
    print(vectorized_datasets['train'][0])

    ###############################################
    if (data_args.output_data_dir is not None):
        vectorized_datasets.save_to_disk(data_args.output_data_dir)
    else:
        storage_options = {"project":"whisper-whispers"}
        vectorized_datasets.save_to_disk("gcs://whisper_bucket2/preprocessed_data/", storage_options=storage_options)

    # 13. Evaluation
    results = {}

    return results


if __name__ == "__main__":
    main()