#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
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
# Modifications made based on code from Hugging Face: Added MoTE module
from abc import ABC, abstractmethod
import logging
import os
import random
import sys
import warnings
from dataclasses import dataclass, field
from typing import List, Optional
from scipy.special import expit
import datasets
import evaluate
import numpy as np
from datasets import Value, load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

import torch
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AffinityPropagation
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

logger = logging.getLogger(__name__)

class Expert(nn.Module):
    def __init__(self, input_dim, num_labels, dropout_prob=0.1):
        super(Expert, self).__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=8)
        self.dense = nn.Linear(2 * input_dim, input_dim)  
        self.dropout = nn.Dropout(dropout_prob)
        self.out_proj = nn.Linear(input_dim, num_labels)
    
    def forward(self, x, cat_token):
        x = self.transformer_layer(x)
        x = x[:, 0, :]  # Take <s> token (equiv. to [CLS])
        x = torch.cat((x, cat_token), dim=-1)  
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Router(ABC, nn.Module):
    """Base Router class"""

    def __init__(self, input_dim, num_experts):
        super(Router, self).__init__()
        self.num_experts = num_experts
        self.weight = nn.Parameter(torch.empty((num_experts, input_dim)))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def gating(self, input: torch.Tensor):
        logits = nn.functional.linear(input, self.weight)
        return logits

    @abstractmethod
    def routing(self, logits: torch.Tensor):
        raise NotImplementedError("Routing function not implemented.")

    @abstractmethod
    def forward(self, input: torch.Tensor):
        raise NotImplementedError("Forward function not implemented.")

class TopKRouter(Router):
    def __init__(self, input_dim, num_experts, topk=2, load_balancing_coef=0.1):
        super(TopKRouter, self).__init__(input_dim, num_experts)
        self.topk = topk
        self.load_balancing_coef = load_balancing_coef

    def aux_loss_load_balancing(self, logits: torch.Tensor):
        """Apply loss-based load balancing to the logits tensor.

        Args:
            logits (torch.Tensor): the logits tensor after gating, shape: [num_tokens, num_experts].

        Returns:
            probs (torch.Tensor): the probabilities tensor after load balancing.
            indices (torch.Tensor): the indices tensor after top-k selection.
        """
        num_tokens = logits.shape[0]
        num_experts = logits.shape[1]
        #pre softmax
        probs = torch.softmax(logits, dim=-1)  #scores = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
        top_probs, top_indices = torch.topk(probs, self.topk, dim=1) 
        
        # TopK without capacity
        tokens_per_expert = torch.bincount(top_indices.view(-1), minlength=num_experts)
       
        # The formula of aux_loss: 
        #aux_loss = sum((probs_per_expert/num_tokens) * (tokens_per_expert/(num_tokens*topk))) * num_experts * moe_aux_loss_coeff.
        aggregated_probs_per_expert = probs.sum(dim=0)
        aux_loss = torch.sum(aggregated_probs_per_expert * tokens_per_expert) * (
            num_experts * self.load_balancing_coef / (num_tokens * num_tokens * self.topk)
        ) #这里 aggregated_probs_per_expert tokens_per_expert都要size是num_experts而不是topk之后的
        return top_probs, top_indices, aux_loss
    
    def routing(self, logits: torch.Tensor):
        scores, indices, load_balancing_loss = self.aux_loss_load_balancing(logits)
        return scores, indices, load_balancing_loss

    def forward(self, input: torch.Tensor):
        logits = self.gating(input)
        logits = logits.view(-1, self.num_experts)
        scores, indices, load_balancing_loss = self.routing(logits)
        return scores, indices, load_balancing_loss
        
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, input_dim)
        self.fc22 = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()
    
    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

def loss_function(recon_x, x, mu, logvar):
    #BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum') 
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD
    # importance 和 load 先不加。。

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_split_ratio: Optional[float] = field(
        default=None, 
        metadata={"help": "Ratio of the training data to be used."}
    )
    eval_split_ratio: Optional[float] = field(
        default=None, 
        metadata={"help": "Ratio of the validation data to be used."}
    )
    do_regression: bool = field(
        default=None,
        metadata={
            "help": "Whether to do regression instead of classification. If None, will be inferred from the dataset."
        },
    )
    text_column_names: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The name of the text column in the input dataset or a CSV/JSON file. "
                'If not specified, will use the "sentence" column for single/multi-label classifcation task.'
            )
        },
    )
    text_column_delimiter: Optional[str] = field(
        default=" ", metadata={"help": "THe delimiter to use to join text columns into a single sentence."}
    )
    train_split_name: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The name of the train split in the input dataset. If not specified, will use the "train" split when do_train is enabled'
        },
    )
    validation_split_name: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The name of the validation split in the input dataset. If not specified, will use the "validation" split when do_eval is enabled'
        },
    )
    test_split_name: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The name of the test split in the input dataset. If not specified, will use the "test" split when do_predict is enabled'
        },
    )
    remove_splits: Optional[str] = field(
        default=None,
        metadata={"help": "The splits to remove from the dataset. Multiple splits should be separated by commas."},
    )
    remove_columns: Optional[str] = field(
        default=None,
        metadata={"help": "The columns to remove from the dataset. Multiple columns should be separated by commas."},
    )
    label_column_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The name of the label column in the input dataset or a CSV/JSON file. "
                'If not specified, will use the "label" column for single/multi-label classifcation task'
            )
        },
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    shuffle_train_dataset: bool = field(
        default=False, metadata={"help": "Whether to shuffle the train dataset or not."}
    )
    shuffle_seed: int = field(
        default=42, metadata={"help": "Random seed that will be used to shuffle the train dataset."}
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
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    metric_name: Optional[str] = field(default=None, metadata={"help": "The metric to use for evaluation."})
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    '''def __post_init__(self):
        if self.dataset_name is None:
            if self.train_file is None or self.validation_file is None:
                raise ValueError(" training/validation file or a dataset name.")

            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."'''
    def __post_init__(self):
        if self.dataset_name is None:
            
            if self.train_file is None:
                raise ValueError("Please provide a training file.")
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."

        if self.validation_file is not None:
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


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
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )


def get_label_list(raw_dataset, split="train") -> List[str]:
    """Get the list of labels from a mutli-label dataset"""

    if isinstance(raw_dataset[split]["label"][0], list):
        label_list = [label for sample in raw_dataset[split]["label"] for label in sample]
        label_list = list(set(label_list))
    else:
        label_list = raw_dataset[split].unique("label")
    # we will treat the label list as a list of string instead of int, consistent with model.config.label2id
    label_list = [str(label) for label in label_list]
    return label_list


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.",
            FutureWarning,
        )
        if model_args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_classification", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
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

    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )
        # Try print some info about the dataset
        logger.info(f"Dataset loaded: {raw_datasets}")
        logger.info(raw_datasets)
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file}
        # Add validation file to data_files if it exists
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        # Get the test dataset: you can provide your own CSV/JSON test file
            
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a dataset name or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            raw_datasets = load_dataset(
                "csv",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
            )
        else:
            # Loading a dataset from local json files
            raw_datasets = load_dataset(
                "json",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
            )


    if data_args.remove_splits is not None:
        for split in data_args.remove_splits.split(","):
            logger.info(f"removing split {split}")
            raw_datasets.pop(split)

    if data_args.train_split_name is not None:
        logger.info(f"using {data_args.validation_split_name} as validation set")
        raw_datasets["train"] = raw_datasets[data_args.train_split_name]
        raw_datasets.pop(data_args.train_split_name)

    if data_args.validation_split_name is not None:
        logger.info(f"using {data_args.validation_split_name} as validation set")
        raw_datasets["validation"] = raw_datasets[data_args.validation_split_name]
        raw_datasets.pop(data_args.validation_split_name)

    if data_args.test_split_name is not None:
        logger.info(f"using {data_args.test_split_name} as test set")
        raw_datasets["test"] = raw_datasets[data_args.test_split_name]
        raw_datasets.pop(data_args.test_split_name)

    if data_args.remove_columns is not None:
        for split in raw_datasets.keys():
            for column in data_args.remove_columns.split(","):
                logger.info(f"removing column {column} from split {split}")
                raw_datasets[split].remove_columns(column)

    if data_args.label_column_name is not None and data_args.label_column_name != "label":
        for key in raw_datasets.keys():
            raw_datasets[key] = raw_datasets[key].rename_column(data_args.label_column_name, "label")

    if data_args.train_split_ratio is not None:
        logger.info(f"Splitting training set with ratio {data_args.train_split_ratio * 100}%.")
        train_validation_split = raw_datasets['train'].train_test_split(test_size=data_args.train_split_ratio, seed=42)
        #train_dataset = train_validation_split['test']
        raw_datasets['train'] = train_validation_split['test']
        if data_args.eval_split_ratio is not None:
            raw_datasets['validation'] = train_validation_split['train'].train_test_split(test_size=data_args.eval_split_ratio, seed=42)['test']

        

    is_regression = (
        raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if data_args.do_regression is None
        else data_args.do_regression
    )

    is_multi_label = False
    if is_regression:
        label_list = None
        num_labels = 1
        # regession requires float as label type, let's cast it if needed
        for split in raw_datasets.keys():
            if raw_datasets[split].features["label"].dtype not in ["float32", "float64"]:
                logger.warning(
                    f"Label type for {split} set to float32, was {raw_datasets[split].features['label'].dtype}"
                )
                features = raw_datasets[split].features
                features.update({"label": Value("float32")})
                try:
                    raw_datasets[split] = raw_datasets[split].cast(features)
                except TypeError as error:
                    logger.error(
                        f"Unable to cast {split} set to float32, please check the labels are correct, or maybe try with --do_regression=False"
                    )
                    raise error

    else:  # classification
        if raw_datasets["train"].features["label"].dtype == "list":  # multi-label classification
            is_multi_label = True
            logger.info("Label type is list, doing multi-label classification")
        # Trying to find the number of labels in a multi-label classification task
        # We have to deal with common cases that labels appear in the training set but not in the validation/test set.
        # So we build the label list from the union of labels in train/val/test.
        label_list = get_label_list(raw_datasets, split="train")
        for split in ["validation", "test"]:
            if split in raw_datasets:
                val_or_test_labels = get_label_list(raw_datasets, split=split)
                diff = set(val_or_test_labels).difference(set(label_list))
                if len(diff) > 0:
                    # add the labels that appear in val/test but not in train, throw a warning
                    logger.warning(
                        f"Labels {diff} in {split} set but not in training set, adding them to the label list"
                    )
                    label_list += list(diff)
        # if label is -1, we throw a warning and remove it from the label list
        for label in label_list:
            if label == -1:
                logger.warning("Label -1 found in label list, removing it.")
                label_list.remove(label)

        label_list.sort()
        num_labels = len(label_list)
        if num_labels <= 1:
            raise ValueError("You need more than one label to do classification.")

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task="text-classification",
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    if is_regression:
        config.problem_type = "regression"
        logger.info("setting problem type to regression")
    elif is_multi_label:
        config.problem_type = "multi_label_classification"
        logger.info("setting problem type to multi label classification")
    else:
        config.problem_type = "single_label_classification"
        logger.info("setting problem type to single label classification")

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    source_model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    ).to(training_args.device)

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # for training ,we will update the config with label infos,
    # if do_train is not set, we will use the label infos in the config
    if training_args.do_train and not is_regression:  # classification, training
        label_to_id = {v: i for i, v in enumerate(label_list)}
        # update config with label infos
        if source_model.config.label2id != label_to_id:
            logger.warning(
                "The label2id key in the model config.json is not equal to the label2id key of this "
                "run. You can ignore this if you are doing finetuning."
            )
        source_model.config.label2id = label_to_id
        source_model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif not is_regression:  # classification, but not training
        logger.info("using label infos in the model config")
        logger.info("label2id: {}".format(source_model.config.label2id))
        label_to_id = source_model.config.label2id
    else:  # regression
        label_to_id = None

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def multi_labels_to_ids(labels: List[str]) -> List[float]:
        ids = [0.0] * len(label_to_id)  # BCELoss requires float as target type
        for label in labels:
            if label not in label_to_id:
                ids[label_to_id[str(label)]] = 1.0
            else:
                ids[label_to_id[label]] = 1.0
        return ids

    def preprocess_function(examples):
        if data_args.text_column_names is not None:
            text_column_names = data_args.text_column_names.split(",")
            # join together text columns into "sentence" column
            examples["sentence"] = examples[text_column_names[0]]
            for column in text_column_names[1:]:
                for i in range(len(examples[column])):
                    examples["sentence"][i] += data_args.text_column_delimiter + examples[column][i]
        # Tokenize the texts
        result = tokenizer(examples["sentence"], padding=padding, max_length=max_seq_length, truncation=True)
        if label_to_id is not None and "label" in examples:
            if is_multi_label:
                result["label"] = [multi_labels_to_ids(l) for l in examples["label"]]
            else:
                result["label"] = [(label_to_id[str(l)] if l != -1 else -1) for l in examples["label"]]
        return result
    # Running the preprocessing pipeline on all the datasets
    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset.")
        train_dataset = raw_datasets["train"]
        if data_args.shuffle_train_dataset:
            logger.info("Shuffling the training dataset")
            train_dataset = train_dataset.shuffle(seed=data_args.shuffle_seed)
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            if "test" not in raw_datasets and "test_matched" not in raw_datasets:
                raise ValueError("--do_eval requires a validation or test dataset if validation is not defined.")
            else:
                logger.warning("Validation dataset not found. Falling back to test dataset for validation.")
                eval_dataset = raw_datasets["test"]
        else:
            eval_dataset = raw_datasets["validation"]

        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict or data_args.test_file is not None:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    if data_args.metric_name is not None:
        metric = (
            evaluate.load(data_args.metric_name, config_name="multilabel")
            if is_multi_label
            else evaluate.load(data_args.metric_name)
        )
        logger.info(f"Using metric {data_args.metric_name} for evaluation.")
    else:
        if is_regression:
            metric = evaluate.load("mse")
            logger.info("Using mean squared error (mse) as regression score, you can use --metric_name to overwrite.")
        else:
            if is_multi_label:
                '''metric = evaluate.load("f1",config_name="multilabel")
                logger.info(
                    "Using multilabel F1 for multi-label classification task, you can use --metric_name to overwrite."
                )'''
                f1_metric = evaluate.load("f1",config_name="multilabel")
                accuracy_metric = evaluate.load("accuracy",config_name="multilabel")
                precision_metric = evaluate.load("precision",config_name="multilabel")
                recall_metric = evaluate.load("recall",config_name="multilabel")
                roc_auc_score = evaluate.load("roc_auc", "multilabel")
                logger.info(
                    "Using multilabel F1, accuracy, precision, recall for multi-label classification task, you can use --metric_name to overwrite."
                )
            else:
                #metric = evaluate.combine("f1","accuracy","precision","recall")
                f1_metric = evaluate.load("f1")
                accuracy_metric = evaluate.load("accuracy")
                precision_metric = evaluate.load("precision")
                recall_metric = evaluate.load("recall")
                roc_auc_score = evaluate.load("roc_auc", "multiclass")
                bi_roc_auc_score = evaluate.load("roc_auc", "binary")
                logger.info("Using F1, accuracy, precision, recall as classification score, you can use --metric_name to overwrite.")

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        if is_regression:
            preds = np.squeeze(preds)
            result = metric.compute(predictions=preds, references=p.label_ids)
        elif is_multi_label:
            pred_probs = expit(preds)
            #print(pred_probs)
            roc_auc_ma=roc_auc_score.compute(references=p.label_ids,prediction_scores=pred_probs,average="macro")["roc_auc"]
            roc_auc_mi=roc_auc_score.compute(references=p.label_ids,prediction_scores=pred_probs,average="micro")["roc_auc"]
            roc_auc_sa=roc_auc_score.compute(references=p.label_ids,prediction_scores=pred_probs,average="samples")["roc_auc"]
            roc_auc_w=roc_auc_score.compute(references=p.label_ids,prediction_scores=pred_probs,average="weighted")["roc_auc"]
            preds = np.array([np.where(p > 0, 1, 0) for p in preds])  # convert logits to multi-hot encoding
            accuracy = accuracy_metric.compute(predictions=preds, references=p.label_ids)["accuracy"]
            samples_f1 = f1_metric.compute(predictions=preds, references=p.label_ids, average="samples")["f1"]
            samples_precision = precision_metric.compute(predictions=preds, references=p.label_ids, average="samples")["precision"]
            samples_recall = recall_metric.compute(predictions=preds, references=p.label_ids, average="samples")["recall"]
            micro_f1 = f1_metric.compute(predictions=preds, references=p.label_ids, average="micro")["f1"]
            micro_precision = precision_metric.compute(predictions=preds, references=p.label_ids, average="micro")["precision"]
            micro_recall = recall_metric.compute(predictions=preds, references=p.label_ids, average="micro")["recall"]
            macro_f1 = f1_metric.compute(predictions=preds, references=p.label_ids, average="macro")["f1"]
            macro_precision = precision_metric.compute(predictions=preds, references=p.label_ids, average="macro")["precision"]
            macro_recall = recall_metric.compute(predictions=preds, references=p.label_ids, average="macro")["recall"]
            weighted_f1 = f1_metric.compute(predictions=preds, references=p.label_ids, average="weighted")["f1"]
            weighted_precision = precision_metric.compute(predictions=preds, references=p.label_ids, average="weighted")["precision"]
            weighted_recall = recall_metric.compute(predictions=preds, references=p.label_ids, average="weighted")["recall"]
            
            result = {
                "accuracy": accuracy,
                "auc_ma":roc_auc_ma,
                "auc_mi":roc_auc_mi,
                "auc_w":roc_auc_w,
                "auc_sa":roc_auc_sa,
                "samples_precision": samples_precision,
                "samples_recall": samples_recall, 
                "samples_f1": samples_f1,
                "micro_precision": micro_precision,
                "micro_recall": micro_recall, 
                "micro_f1": micro_f1,
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "macro_f1": macro_f1,
                "weighted_precision":weighted_precision,
                "weighted_recall":weighted_recall,
                "weighted_f1":weighted_f1,
            }
        else:
            
            if num_labels == 2:
                preds = np.argmax(preds, axis=1)
                accuracy = accuracy_metric.compute(predictions=preds, references=p.label_ids)["accuracy"]
                binary_precision = precision_metric.compute(predictions=preds, references=p.label_ids, average="binary")["precision"]
                binary_recall = recall_metric.compute(predictions=preds, references=p.label_ids, average="binary")["recall"]
                binary_f1 = f1_metric.compute(predictions=preds, references=p.label_ids, average="binary")["f1"]
                result = {
                    "accuracy": accuracy,
                    "precision": binary_precision,
                    "recall": binary_recall,
                    "f1": binary_f1,
                }
            else:
                preds_tensor = torch.tensor(preds)
                probs = nn.functional.softmax(preds_tensor, dim=1)
                macro_ovr_auc = roc_auc_score.compute(references=p.label_ids,prediction_scores=probs,  average="macro",multi_class='ovr')["roc_auc"]
                macro_ovo_auc = roc_auc_score.compute(references=p.label_ids,prediction_scores=probs,  average="macro",multi_class='ovo')["roc_auc"]
                weighted_ovr_auc = roc_auc_score.compute(references=p.label_ids,prediction_scores=probs,  average="weighted",multi_class='ovr')["roc_auc"]
                weighted_ovo_auc = roc_auc_score.compute(references=p.label_ids,prediction_scores=probs,  average="weighted",multi_class='ovo')["roc_auc"]
                preds = np.argmax(preds, axis=1)    
                accuracy = accuracy_metric.compute(predictions=preds, references=p.label_ids)["accuracy"]
                micro_f1 = f1_metric.compute(predictions=preds, references=p.label_ids, average="micro")["f1"]
                micro_precision = precision_metric.compute(predictions=preds, references=p.label_ids, average="micro")["precision"]
                micro_recall = recall_metric.compute(predictions=preds, references=p.label_ids, average="micro")["recall"]
                macro_f1 = f1_metric.compute(predictions=preds, references=p.label_ids, average="macro")["f1"]
                macro_precision = precision_metric.compute(predictions=preds, references=p.label_ids, average="macro")["precision"]
                macro_recall = recall_metric.compute(predictions=preds, references=p.label_ids, average="macro")["recall"]
                weighted_f1 = f1_metric.compute(predictions=preds, references=p.label_ids, average="weighted")["f1"]
                weighted_precision = precision_metric.compute(predictions=preds, references=p.label_ids, average="weighted")["precision"]
                weighted_recall = recall_metric.compute(predictions=preds, references=p.label_ids, average="weighted")["recall"]
                result = {
                    "accuracy": accuracy,
                    "micro_precision": micro_precision,
                    "micro_recall": micro_recall, 
                    "micro_f1": micro_f1,
                    "macro_precision": macro_precision,
                    "macro_recall": macro_recall,
                    "macro_f1": macro_f1,
                    "weighted_precision":weighted_precision,
                    "weighted_recall":weighted_recall,
                    "weighted_f1":weighted_f1,
                    "macro_ovr_auc":macro_ovr_auc,
                    "macro_ovo_auc":macro_ovo_auc,
                    "weighted_ovr_auc":weighted_ovr_auc,
                    "weighted_ovo_auc":weighted_ovo_auc,
                }
        return result
    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None
    # --------------------------------------------------------------
    #                    clustering
    # --------------------------------------------------------------       
    # input_ids 和 attention_mask
    train_input_ids = torch.tensor(train_dataset["input_ids"])
    train_attention_mask = torch.tensor(train_dataset["attention_mask"])
    train_label= torch.tensor(train_dataset["label"])

    train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_label)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=False)
    test_input_ids = torch.tensor(predict_dataset["input_ids"])
    test_attention_mask = torch.tensor(predict_dataset["attention_mask"])
    test_label= torch.tensor(predict_dataset["label"])
    test_dataset = TensorDataset(test_input_ids, test_attention_mask, test_label)
    test_dataloader = DataLoader(test_dataset, batch_size=8)
    cls_tokens_train = []   

    source_model.eval()
    with torch.no_grad():
        for batch in train_dataloader:
            batch_input_ids, batch_attention_mask, _ = [b.to(training_args.device) for b in batch]
            outputs = source_model.roberta(batch_input_ids, attention_mask=batch_attention_mask)
        
            cls_token = outputs.last_hidden_state[:, 0, :]  
            cls_tokens_train.append(cls_token.cpu().numpy())

    cls_tokens_train = np.concatenate(cls_tokens_train, axis=0)
    cls_tokens_train = torch.tensor(cls_tokens_train, dtype=torch.float)
    train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_label, cls_tokens_train)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=False)

    k = 4  
    kmeans = KMeans(n_clusters=k, random_state=0).fit(cls_tokens_train)
    cluster_labels = kmeans.labels_
    cluster_centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float).to(training_args.device)
    cluster_indices = [np.where(cluster_labels == i)[0] for i in range(k)]
    unique, counts = np.unique(cluster_labels, return_counts=True)
    print(f"Cluster distribution: {dict(zip(unique, counts))}")
    for i in range(k):
        print(f'Cluster {i}: {counts[i] / len(cluster_labels):.2%}')

#------------------------------------------------------
        # router warmup
#------------------------------------------------------
    input_dim = cls_tokens_train.shape[1]
    num_experts=k
    topk=2
    load_balancing_coef=0.1
    dropout_prob = 0.1

    experts = [Expert(input_dim, num_labels, dropout_prob).to(training_args.device) for _ in range(k)]
    router = TopKRouter(input_dim, num_experts, topk, load_balancing_coef).to(training_args.device)
    optimizers = [optim.AdamW(expert.parameters(), lr=1e-4) for expert in experts]
    router_optimizer = optim.AdamW(router.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 20    
    # Warm-up Router
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_idx, batch in enumerate(train_dataloader):
            batch_input_ids, batch_attention_mask, _ , _ = [b.to(training_args.device) for b in batch]
                  
            batch_start_idx = batch_idx * train_dataloader.batch_size
            batch_end_idx = batch_start_idx + len(batch_input_ids)
            batch_cluster_labels = torch.tensor(cluster_labels[batch_start_idx:batch_end_idx], dtype=torch.long).to(training_args.device)
            
            outputs = source_model.roberta(batch_input_ids, attention_mask=batch_attention_mask).last_hidden_state
            gating_scores = router.gating(outputs[:, 0, :])
            
            router_optimizer.zero_grad()

            loss_router = criterion(gating_scores, batch_cluster_labels)
            loss_router.backward()
            router_optimizer.step()
            total_loss += loss_router.item()
            
        avg_loss = total_loss / len(train_dataloader)
        print(f"Warmup Router Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        
    # Training Experts with router
    num_epochs = 20
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in train_dataloader:
            batch_input_ids, batch_attention_mask, batch_labels, _ = [b.to(training_args.device) for b in batch]
            outputs = source_model.roberta(batch_input_ids, attention_mask=batch_attention_mask).last_hidden_state
            scores, indices, load_balancing_loss = router(outputs[:, 0, :]) 
            
            batch_size = outputs.size(0)
            cat_tokens = torch.zeros((batch_size, input_dim), device=training_args.device)
            for i in range(batch_size):
                expert_idx = indices[i, 0].item()  
                cat_tokens[i, :] = outputs[i, 0, :] - cluster_centroids[expert_idx]

            final_loss = 0.0
            for i in range(indices.size(1)):  # iterate over top-k experts
                for j in range(indices.size(0)):  # iterate over batch size
                    expert_idx = indices[j, i].item()  # get the expert index for each instance in the batch
                    expert = experts[expert_idx]  # use the integer value as index
                    expert_optimizer = optimizers[expert_idx]                 
                    expert_optimizer.zero_grad()
                    predictions = expert(outputs[j:j+1], cat_tokens[j:j+1])  # select the j-th instance
                    if config.problem_type == "regression":
                        loss_fct = nn.MSELoss()
                        if num_labels == 1:
                            loss = loss_fct(predictions.squeeze(), batch_labels[j:j+1].squeeze())
                        else:
                            loss = loss_fct(predictions, batch_labels[j:j+1]) #predictions is logits
                    elif config.problem_type == "single_label_classification":
                        loss_fct = nn.CrossEntropyLoss()
                        loss = loss_fct(predictions, batch_labels[j:j+1])
                    elif config.problem_type == "multi_label_classification":
                        loss_fct = nn.BCEWithLogitsLoss()                      
                        loss = loss_fct(predictions, batch_labels[j:j+1].float())
                   
                    final_loss += loss
            
            final_loss = final_loss / topk + load_balancing_loss            
            final_loss.backward()
            
            for i in range(indices.size(1)):
                for j in range(indices.size(0)):
                    expert_idx = indices[j, i].item()
                    expert_optimizer = optimizers[expert_idx]
                    expert_optimizer.step()
            
            total_loss += final_loss.item()
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_loss:.4f}")
        model_save_path = f"mimic_bceloss_concatenated_expert_{i}_epoch_{epoch+1}.pt"
    torch.save(experts[i].state_dict(), model_save_path)
    # Using router for prediction
    test_predictions = []
    test_labels = []
    test_logits = []
    with torch.no_grad():
        for batch in test_dataloader:
            batch_input_ids, batch_attention_mask, batch_labels = [b.to(training_args.device) for b in batch]
            outputs = source_model.roberta(batch_input_ids, attention_mask=batch_attention_mask).last_hidden_state
            scores, indices, _ = router(outputs[:, 0, :])           
            batch_size = outputs.size(0)
            cat_tokens = torch.zeros((batch_size, input_dim), device=training_args.device)
            for i in range(batch_size):
                expert_idx = indices[i, 0].item()  
                cat_tokens[i, :] = outputs[i, 0, :] - cluster_centroids[expert_idx]                    
            expert_logits = np.zeros((batch_size, topk, num_labels))           
            for i in range(indices.size(1)):  # iterate over top-k experts
                for j in range(indices.size(0)):  # iterate over batch size
                    expert_idx = indices[j, i].item()  # get the expert index for each instance in the batch
                    logits = experts[expert_idx](outputs[j:j+1], cat_tokens[j:j+1])  # select the j-th instance
                    expert_logits[j, i] = logits.cpu().numpy()
          
            weighted_sum = np.sum(scores[:, :, None].cpu().numpy() * expert_logits, axis=1)          
            logits = weighted_sum 
            
            if is_regression:
                predictions = np.squeeze(logits)
            elif is_multi_label:
                predictions = np.array([np.where(p > 0, 1, 0) for p in logits])
            else:
                predictions = np.argmax(logits, axis=1)           
            test_logits.extend(logits)
            test_predictions.extend(predictions)
            test_labels.extend(batch_labels.cpu().numpy())
      
    eval_preds = EvalPrediction(predictions=np.array(test_logits), label_ids=np.array(test_labels))
    metrics = compute_metrics(eval_preds)
    print(f'Test Metrics: {metrics}')

    output_dir = training_args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_predict_file = os.path.join(training_args.output_dir, "predict_results.txt")
    with open(output_predict_file, "w") as writer:
        logger.info("***** Predict results *****")
        writer.write("index\tprediction\tlabel\n")
        for index, (prediction, label) in enumerate(zip(test_predictions, test_labels)):
            writer.write(f"{index}\t{prediction}\t{label}\n")
    logger.info("Predict results saved at {}".format(output_predict_file))

    output_metrics_file = os.path.join(training_args.output_dir, "metrics.json")
    with open(output_metrics_file, "w") as writer:
        json.dump(metrics, writer, indent=4)
    logger.info("Metrics saved at {}".format(output_metrics_file))

if __name__ == "__main__":
    main()