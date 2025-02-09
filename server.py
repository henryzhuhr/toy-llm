from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

def main():
    model_name_or_path=".cache/models/Qwen2.5-0.5B-Instruct"
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # Pass the default decoding hyperparameters of Qwen2.5-7B-Instruct
    # max_tokens is for the maximum length for generation.
    sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)

    # Input the model name or path. Can be GPTQ or AWQ models.
    llm = LLM(model=model_name_or_path)

    # Prepare your prompts
    prompt = "Tell me something about large language models."
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # generate outputs
    outputs = llm.generate([text], sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        
if __name__ == "__main__":
    main()
