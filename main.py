import argparse

import torch
from transformers import pipeline
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    models,
    pipeline,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
import time
from functools import wraps
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Args:
    def __init__(self):
        args = self.get_args()
        self.weight: str = args.weight

    @staticmethod
    def get_args():
        parser = argparse.ArgumentParser(description="Process some integers.")
        parser.add_argument(
            "--weight",
            type=str,
            default=".cache/DeepSeek-R1-Distill-Qwen-1.5B",
            help="Path to the model",
        )
        args = parser.parse_args()
        return args


def main():
    args = Args()

    # 加载 tokenizer 和 model
    tokenizer = AutoTokenizer.from_pretrained(args.weight)
    model: models.qwen2.Qwen2ForCausalLM = AutoModelForCausalLM.from_pretrained(
        args.weight
    )
    print(type(model))
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        model.to(device)
        print("MPS is available")
    print("加载模型成功")

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=model.device,
    )
    # input_text = "你是谁"
    # messages = [
    #     {
    #         "role": "user",
    #         "content": f"Think in English and response in Chinese:\n{input_text}",
    #     },
    # ]

    # outputs = generator(
    #     messages,
    #     max_new_tokens=1024,
    #     do_sample=True,
    #     temperature=0.7,
    #     pad_token_id=tokenizer.pad_token_id,
    # )
    # print(outputs[-1]["generated_text"][-1]["content"])

    messages = [
        {
            "role": "system",
            "content": "您是一个有帮助的、专注的、直截了当的助手。",
        },
        {
            "role": "user",
            # "content": "您可以提供香蕉和火龙果的食用搭配方法吗？",
            "content": "你是谁",
        },
    ]
    # prompt = tokenizer.apply_chat_template(messages,
    #                                             tokenize=False,
    #                                             add_generation_prompt=True)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    generation_args = {
        "max_new_tokens": 3072,
        "return_full_text": False,
        "temperature": 0.0,
        "do_sample": False,
    }

    # inputs = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt").to("cuda")
    # outputs = model.generate(input_ids=inputs.to(model.device),
    #                          max_new_tokens=max_length,
    #                          do_sample=True,
    #                          temperature=0.1,
    #                          top_k=50,
    #                          )
    output = pipe(messages, **generation_args)
    print(output[0]["generated_text"])


def stop_watch(func):
    @wraps(func)
    def wrapper(*args, **kargs):
        logger.debug(f"🚦 [@stop_watch] measure time to run `{func.__name__}`.")
        start = time.time()
        result = func(*args, **kargs)
        elapsed_time = time.time() - start
        logger.debug(
            f"🚦 [@stop_watch] take {elapsed_time:.3f} sec to run `{func.__name__}`."
        )
        return result

    return wrapper


if __name__ == "__main__":
    main()
