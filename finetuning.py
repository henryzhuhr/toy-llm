# pip install GPUtil

import os

import pandas as pd
import transformers

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set to the GPU ID (0 for T4)
import argparse
import torch
import GPUtil
import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
from datasets import Dataset
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datetime import datetime
from datasets import load_dataset


def main():
    # Get the arguments or load from .env
    args = Args()

    print(GPUtil.showUtilization())

    if torch.cuda.is_available():
        print("GPU is available!")
    else:
        print("GPU not available.")

    # Load the dataset
    dataset = _load_dataset()

    # Load the model
    model, tokenizer = _load_model(args.model_id)

    # Train the model
    train(model, tokenizer, dataset)


def _load_dataset(dataset_id: str = "HuggingFaceH4/helpful-instructions"):
    dataset = load_dataset("HuggingFaceH4/helpful-instructions")
    print(pd.DataFrame(dataset["train"]))
    return dataset


def _load_model(pretrained_model_name_or_path):
    bnb_config = BitsAndBytesConfig(  # pip install bitsandbytes
        load_in_4bit=True,  # Enables loading the model using 4-bit quantization, reducingmemory and computational costs.
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # Sets the computational data type for the 4-bit quantized model, controlling precision during inference or training.
    )

    """
    Load the model from local or Hugging Face
    """
    # Load the model
    model: Qwen2ForCausalLM = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto",  # Helps with memory management
    )
    model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

    # Load the tokenizer
    tokenizer: Qwen2Tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        use_fast=False,
        trust_remote_code=True,
        add_eos_token=True,
    )
    print(type(tokenizer))

    """
    the End-of-sentence (eos) token and the padding (pad) token.
    """

    if tokenizer.pad_token is None:
        print("Adding the pad token [{[PAD]}] to the tokenizer")
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    else:
        print(f"Pad token [{tokenizer.pad_token}] already exists in the tokenizer. ")

    # set the pad token to indicate that it's the end-of-sentence
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def train(model: Qwen2ForCausalLM, tokenizer: Qwen2Tokenizer, dataset):
    OUTPUT_DIR = f"experiments-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

    # training arguments
    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=10,
        learning_rate=2e-4,
        fp16=True,
        save_total_limit=3,
        logging_steps=1,
        output_dir=OUTPUT_DIR,
        max_steps=200,  # try more steps if you can
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        report_to="tensorboard",
    )
    trainer = transformers.Trainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        data_collator=transformers.DataCollatorForLanguageModeling(
            tokenizer, mlm=False
        ),
    )

    pass


class TrainDataset:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path


class Args:
    """命令行参数"""

    def __init__(self):
        args = self.get_args()
        self.model_id: str = args.model_id

    @staticmethod
    def get_args():
        try:
            from dotenv import load_dotenv

            load_dotenv()  # 打印所有的环境变量 print(json.dumps(dict(os.environ), indent=4))
        except ImportError as e:
            print(
                "dotenv not installed, skipping. "
                "Install with `pip install python-dotenv`"
            )

        parser = argparse.ArgumentParser(description="Process some integers.")
        parser.add_argument(
            "--model-id",
            type=str,
            default=os.getenv("MODEL_PATH"),
            help="pretrained model name or path",
        )
        return parser.parse_args()


if __name__ == "__main__":
    main()
