import os
import sys
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch

from datasets import Dataset, DatasetDict
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    AutoModelForSequenceClassification,
    HfArgumentParser,
    DataCollatorWithPadding,
    set_seed,
)
import evaluate
import peft
import adapters
import wandb
import random

# Try relative import for test
try:
    from .dataHelper import get_dataset
except ImportError:
    from dataHelper import get_dataset


@dataclass
class BaseArgs:
    '''Base arguments for the training script.'''

    dataset: str = field(metadata={"help": "Dataset name"})
    model_name: str = field(metadata={"help": "Model name or path"})
    sep_token: str = field(default="<SEP>", metadata={"help": "Separator token"})
    peft: Optional[str] = field(default=None, metadata={"help": "PEFT method: 'lora' or 'adapter'"})
    max_length: int = field(default=256, metadata={"help": "Maximum sequence length"})


@dataclass
class AdapterArgs:
    '''Arguments for Bottleneck Adapter.'''
    
    adapter_size: int = field(default=64, metadata={"help": "Adapter size"})
    adapter_dropout: float = field(default=0.1, metadata={"help": "Adapter dropout"})


@dataclass
class LoraArgs:
    '''Arguments for LoRA.'''

    rank: int = field(default=16, metadata={"help": "LoRA rank"})
    alpha: int = field(default=32, metadata={"help": "LoRA alpha"})
    dropout: float = field(default=0.1, metadata={"help": "LoRA dropout"})


logger = logging.getLogger(__name__)


def print_trainable_parameters(model):
    """
    Print out the number of trainable parameters in the model.
    transformers models have a `print_trainable_parameters` method, but not all models have it.
    """

    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params:,} || "
          f"all params: {all_params:,} || "
          f"trainable%: {100 * trainable_params / all_params:.2f}%")


def parse_arguments():
    '''Parse command line arguments into dataclasses.'''

    parser = HfArgumentParser((BaseArgs, TrainingArguments, AdapterArgs, LoraArgs))
    base_args, train_args, adapter_args, lora_args = parser.parse_args_into_dataclasses()
    return base_args, train_args, adapter_args, lora_args


def set_random_seed(seed: int) -> None:
    '''Set random seed for training.'''

    set_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_logger() -> None:
    '''Set up the logger to print messages to stdout.'''

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info(f"Logging level: {logging.getLevelName(logger.getEffectiveLevel())}")


def load_data(dataset_name: str,
              sep_token: str = "<SEP>") -> Tuple[DatasetDict, int]:
    '''Load dataset and return the number of labels.'''

    raw_dataset = get_dataset(dataset_name, sep_token)
    num_labels = len(set(raw_dataset['train']['label']))
    return raw_dataset, num_labels


def get_model(model_name: str, num_labels: int):
    '''Load model and tokenizer.'''

    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = num_labels
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Handle pad_token for models that don't have one
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        config=config,
        ignore_mismatched_sizes=True
    )
    
    # Resize token embeddings if we added a pad token
    if tokenizer.pad_token_id is None or model.config.pad_token_id is None:
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id
    
    return tokenizer, model


def tokenize_data(raw_dataset, tokenizer, max_length: int = 256):
    '''Tokenize the dataset.'''

    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            max_length=max_length,
            padding=False,
            truncation=True
        )
    
    # Remove only 'text' column, keep 'label'
    columns_to_remove = ['text']
    
    train_dataset = raw_dataset['train'].map(
        tokenize_function,
        batched=True,
        remove_columns=columns_to_remove
    )
    eval_dataset = raw_dataset['test'].map(
        tokenize_function,
        batched=True,
        remove_columns=columns_to_remove
    )
    
    return train_dataset, eval_dataset


def get_lora_model(model, lora_args):
    '''Initialize the model with LoRA modules.'''

    from peft import LoraConfig, get_peft_model, TaskType
    
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_args.rank,
        lora_alpha=lora_args.alpha,
        lora_dropout=lora_args.dropout,
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def get_adapter_model(model: AutoModelForSequenceClassification, adapter_args):
    '''Initialize the model with Bottleneck Adapter modules.'''

    from adapters import BnConfig
    
    adapter_config = BnConfig(
        mh_adapter=True,
        output_adapter=True,
        reduction_factor=adapter_args.adapter_size,
        non_linearity="relu",
    )
    
    # Init the model with adapter
    adapters.init(model)
    # Add adapter to the model
    model.add_adapter("task_adapter", config=adapter_config)
    model.train_adapter("task_adapter")
    model.set_active_adapters("task_adapter")
    
    print_trainable_parameters(model)
    return model


def get_data_collator(tokenizer):
    '''Define data collator for padding.'''

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return data_collator


def compute_metrics(eval_pred) -> Dict[str, float]:
    '''Compute accuracy, macro F1 and micro F1.'''

    # Handle both tuple format and EvalPrediction object format
    if hasattr(eval_pred, 'predictions'):
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids if hasattr(eval_pred, 'label_ids') else eval_pred.labels
    else:
        predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    predictions = np.argmax(predictions, axis=1)
    
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    macro_f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")["f1"]
    micro_f1 = f1_metric.compute(predictions=predictions, references=labels, average="micro")["f1"]
    weighted_f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")["f1"]
    
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "weighted_f1": weighted_f1
    }


def get_trainer(
    model,
    train_args,
    train_dataset,
    eval_dataset,
    tokenizer,
    data_collator,
    compute_metrics_fn
):
    '''Define Trainer for training and evaluation.'''
    
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
    )
    return trainer


def main():
    # Parse arguments
    base_args, train_args, adapter_args, lora_args = parse_arguments()

    # Set seed before initializing model.
    set_random_seed(train_args.seed)

    # Set up logging
    set_logger()

    # load dataset
    raw_dataset, num_labels = load_data(base_args.dataset, base_args.sep_token)

    # get model and tokenizer
    tokenizer, model = get_model(base_args.model_name, num_labels)

    # tokenize dataset
    train_dataset, eval_dataset = tokenize_data(raw_dataset, tokenizer,
                                              base_args.max_length)

    # peft method
    if base_args.peft is not None:
        if base_args.peft.lower() == "lora":
            model = get_lora_model(model, lora_args)
        elif base_args.peft.lower() == "adapter":
            model = get_adapter_model(model, adapter_args)
        else:
            raise ValueError("Unsupported PEFT method!")

    data_collator = get_data_collator(tokenizer)
    if base_args.peft is not None:
        if base_args.peft.lower() == "lora":
            run_name = f"{base_args.model_name.split('/')[-1]}_{base_args.dataset}_{base_args.peft}_{lora_args.rank}_{lora_args.alpha}_{train_args.seed}"
        elif base_args.peft.lower() == "adapter":
            run_name = f"{base_args.model_name.split('/')[-1]}_{base_args.dataset}_{base_args.peft}_{adapter_args.adapter_size}_{train_args.seed}"
        else:
            raise ValueError("Unsupported PEFT method!")
    else:
        run_name = f"{base_args.model_name.split('/')[-1]}_{base_args.dataset}_{train_args.seed}"

    # Initialize wandb
    wandb.init(
        project="hw2-huggingface",
        name=run_name,
        config={
            "dataset": base_args.dataset,
            "model_name": base_args.model_name,
            "peft": base_args.peft,
            "epochs": train_args.num_train_epochs,
            "batch_size": train_args.per_device_train_batch_size,
            "lr": train_args.learning_rate,
            "max_length": base_args.max_length,
        }
    )

    # Create trainer
    trainer = get_trainer(
        model=model,
        train_args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics_fn=compute_metrics,
    )

    if train_args.do_train:
        logger.info("*** Train ***")
        train_result = trainer.train()
        
        # Save model
        trainer.save_model()
        trainer.save_state()
        
        logger.info(f"Training completed. Loss: {train_result.training_loss}")
        wandb.log({"train_loss": train_result.training_loss})

    if train_args.do_eval:
        logger.info("*** Evaluation ***")
        metrics = trainer.evaluate()
        
        logger.info(f"Evaluation metrics: {metrics}")
        wandb.log(metrics)

    if train_args.do_predict:
        logger.info("*** Predict ***")
        predictions = trainer.predict(eval_dataset)
        pred_labels = np.argmax(predictions.predictions, axis=1)
        
        # Save predictions
        with open("predict_results.txt", "w") as f:
            for label in pred_labels:
                f.write(f"{label}\n")
        
        logger.info(f"Predictions saved to predict_results.txt")
    
    return
    wandb.finish()


if __name__ == "__main__":
    main()
