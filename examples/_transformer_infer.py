import argparse
from threading import Thread
from typing import List
import torch
from torch import nn
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedModel,
    AutoModelForCausalLM,
    models,
    TextIteratorStreamer,
)


def main():
    args = Args()

    # 加载 tokenizer 和 model
    tokenizer = AutoTokenizer.from_pretrained(args.weight)
    model: models.qwen2.Qwen2ForCausalLM = AutoModelForCausalLM.from_pretrained(
        args.weight,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        # torch_dtype="auto",
        device_map="auto",
    )

    _demo_chat(model, tokenizer)
    return

    prompts = [
        "你好，你是谁？",
        "你的知识储备到哪一年？",
        "Deepseek是什么？",
        "怎么样评价这一家公司",
    ]
    history = [
        {
            "role": "system",
            "content": "您是一个有用的助手。",
        }
    ]
    history = []
    response = ""
    for prompt in prompts:
        print()
        print("❓ prompt  : ", prompt)
        print("🤖 response:")
        partial_text = ""
        for new_text in _chat_stream(
            model,
            tokenizer,
            prompt,
            history,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        ):
            print(new_text, end="", flush=True)
            partial_text += new_text
        response = partial_text

        history.append([prompt, response])
    print()


def _demo_chat(
    model: models.qwen2.Qwen2ForCausalLM,
    # model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
):
    """
    推理 demo
    """
    from transformers.modeling_outputs import CausalLMOutputWithPast

    prompt = {
        "role": "user",
        "content": "你好，你是谁？",
    }
    # 输入文本 应用模板
    input_text = tokenizer.apply_chat_template(
        [prompt],
        add_generation_prompt=True,
        tokenize=False,
    )
    print(
        "input_text:", input_text
    )  # input_text: <｜begin▁of▁sentence｜>你好，你是谁？<｜Assistant｜>

    # 输入文本编码
    inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
    print("inputs:", inputs)
    print("inputs:", inputs["input_ids"].shape)  # torch.Size([1, N])

    # 模型推理
    with torch.no_grad():
        outputs: CausalLMOutputWithPast = model.forward(**inputs)
    print("outputs:", outputs)
    print("outputs:", outputs.logits.shape)  # torch.Size([1, N, 151936])

    # 解码
    generated_text = tokenizer.decode(outputs.logits[0].argmax(dim=-1))
    print("generated_text:", generated_text)

    # 调用 model.generate 方法生成文本
    generation_kwargs = {
        **inputs,
        "max_new_tokens": 200,
        "temperature": 0.1,
        "do_sample": True,
        "top_p": 1.0,
        "top_k": 50,
        "repetition_penalty": 1.1,
    }
    with torch.no_grad():
        output = model.generate(**generation_kwargs)
    print("outputs:", outputs)
    print("outputs:", outputs.logits.shape)  # torch.Size([1, N, 151936])

    # 解码生成的输出
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("generated_text:", generated_text)


def _chat_stream(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    query,
    history: List[List[str]],
    max_new_tokens=200,
    temperature=0.2,
):
    """
    Chat with the model, streaming the output as it is generated.
    """
    conversation = []
    for query_h, response_h in history:
        conversation.append({"role": "user", "content": query_h})
        conversation.append({"role": "assistant", "content": response_h})
    conversation.append({"role": "user", "content": query})
    input_text = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(
        tokenizer=tokenizer, skip_prompt=True, timeout=60.0, skip_special_tokens=True
    )
    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": True,
        "top_p": 1.0,
        "top_k": 50,
        "repetition_penalty": 1.1,
    }
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for new_text in streamer:
        yield new_text


class Args:
    """命令行参数"""

    def __init__(self):
        args = self.get_args()
        self.weight: str = args.weight

        # inference parameters
        self.max_new_tokens: int = args.max_new_tokens
        self.temperature: float = args.temperature

    @staticmethod
    def get_args():
        parser = argparse.ArgumentParser(description="Process some integers.")
        parser.add_argument(
            "--weight",
            type=str,
            default=".cache/models/DeepSeek-R1-Distill-Qwen-1.5B",
            # default=".cache/models/Qwen2.5-1.5B-Instruct",
            help="Path to the model",
        )
        # inference parameters
        parser.add_argument("--max-new-tokens", type=int, default=4096)
        parser.add_argument("--temperature", type=float, default=0.2)
        args = parser.parse_args()
        return args


if __name__ == "__main__":
    main()
