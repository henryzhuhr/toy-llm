from openai import OpenAI


def main():
    client = OpenAI(
        base_url="http://ollama-server:11434/v1/",
        api_key="ollama",  # required but ignored
    )
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "说这是一个测试",
            }
        ],
        model="qwen2.5:3b",
    )

    print(chat_completion)


if __name__ == "__main__":
    main()
