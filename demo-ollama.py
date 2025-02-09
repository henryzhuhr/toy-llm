import argparse
import os
from ollama import chat
from ollama import ChatResponse


def main():
    args = Args()

    model_name_or_path = args.model_id
    print(model_name_or_path)
    
    try:
        # 获取软链接指向的绝对路径
        target_path = os.readlink(model_name_or_path)
        if not os.path.isabs(target_path):
            # 如果返回的路径是相对路径，将其转换为绝对路径
            base_dir = os.path.dirname(os.path.abspath(model_name_or_path))
            target_path = os.path.join(base_dir, target_path)
        
        print(f"软链接 {model_name_or_path} 指向的原始绝对路径是: {target_path}")
    except FileNotFoundError:
        print(f"未找到软链接文件 {model_name_or_path}。")
    except OSError as e:
        print(f"读取软链接时发生错误: {e}")
        
    response: ChatResponse = chat(
        model=target_path,
        messages=[
            {
                "role": "user",
                "content": "Why is the sky blue?",
            },
        ],
    )
    print(response["message"]["content"])
    # or access fields directly from the response object
    print(response.message.content)


class Args:
    """命令行参数"""

    def __init__(self):
        args = self.get_args()
        self.model_id: str = args.model_id

        # inference parameters
        self.max_new_tokens: int = args.max_new_tokens
        self.temperature: float = args.temperature

    @staticmethod
    def get_args():
        parser = argparse.ArgumentParser(description="Process some integers.")
        parser.add_argument(
            "--model-id",
            type=str,
            default=os.getenv("MODEL_PATH"),
            help="pretrained model name or path",
        )
        # inference parameters
        parser.add_argument("--max-new-tokens", type=int, default=4096)
        parser.add_argument("--temperature", type=float, default=0.2)
        args = parser.parse_args()
        return args


if __name__ == "__main__":
    main()
