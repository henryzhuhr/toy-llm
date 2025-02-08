import argparse
from threading import Thread
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    models,
    TextIteratorStreamer,
)


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
            # default=".cache/DeepSeek-R1-Distill-Qwen-7B",
            default=".cache/Qwen2.5-0.5B-Instruct",
            help="Path to the model",
        )
        args = parser.parse_args()
        return args


def main():
    args = Args()

    # 加载 tokenizer 和 model
    tokenizer = AutoTokenizer.from_pretrained(args.weight)
    model: models.qwen2.Qwen2ForCausalLM = AutoModelForCausalLM.from_pretrained(
        args.weight,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        # torch_dtype=torch.float16,
        torch_dtype="auto",
        device_map="auto",
    )
    print(type(model))
    # if torch.backends.mps.is_available():
    #     device = torch.device("mps")
    #     model.to(device)
    #     print("MPS is available")
    print("加载模型成功")

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
    history, response = [], ""
    for prompt in prompts:
        print()
        print("❓ prompt: ", prompt)
        print("✅ response:")
        partial_text = ""
        for new_text in _chat_stream(model, tokenizer, prompt, history):
            print(new_text, end="", flush=True)
            partial_text += new_text
        response = partial_text

        history.append((prompt, response))


def _chat_stream(model, tokenizer, query, history):
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
        "max_new_tokens": 2048,
        "temperature": 0.1,
        "do_sample": True,
        "top_p": 1.0,
        "top_k": 50,
        "repetition_penalty": 1.1,
    }
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for new_text in streamer:
        yield new_text


if __name__ == "__main__":
    main()
