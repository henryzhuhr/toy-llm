import json
import os
import pandas as pd
from ollama import Client

# 定义文件路径
CSV_FILE = ""

TOOLS = []


def main():

    client = Client(
        # host="http://ollama-server:11434",
        host="http://localhost:11434",
        headers={"x-some-header": "some-value"},
    )

    # 读取 CSV 文件
    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        print(f"错误：文件 {CSV_FILE} 未找到，请检查路径是否正确。")
        return

    # 检查是否包含 deepseek 列
    if "deepseek" not in df.columns:
        print("错误：CSV 文件中没有找到 'deepseek' 列")
        return
    print(df.columns)

    # 反转 deepseek 列中的每个字符串
    # df["deepseek"] = df["deepseek"].apply(lambda x: process_item(client, x))
    # df["deepseek"] = process_item(client, df["question"])
    for index, row in df.iterrows():
        print(index, row["question"])
        df.at[index, "deepseek"] = process_item(client, row["question"])

    # 保存为新的 CSV 文件
    try:
        save_file_name, save_file_suffix = os.path.splitext(CSV_FILE)
        save_file_path = f"{save_file_name}-new{save_file_suffix}"
        df.to_csv(save_file_path, index=False, encoding="utf-8")
        print(f"处理完成，已保存到 {save_file_path}")
    except Exception as e:
        print(f"保存文件时出错：{e}")


def process_item(client: Client, question: str):
    messages = [
        {
            "role": "user",
            "content": question,
        },
    ]
    model_name = "qwen2.5:7b"
    response = client.chat(
        model=model_name,
        messages=messages,
        tools=TOOLS,  # 输入工具，会构造成工具调用
    )
    tool_call_results = []
    if tool_calls := response["message"].get("tool_calls", None):
        for tool_call in tool_calls:

            # print("🟢 ", tool_call)
            if fn_call := tool_call.get("function"):
                fn_name: str = fn_call["name"]
                fn_args: dict = fn_call["arguments"]
                for k in list(fn_args.keys()):
                    if fn_args[k] in ["", None]:
                        del fn_args[k]
                    elif isinstance(fn_args[k], list) and len(fn_args[k]) == 0:
                        del fn_args[k]
                tool_call_result = {"name": fn_name, "fn_args": fn_args}
                print("🟢 ", json.dumps(tool_call_result, ensure_ascii=False))
                tool_call_results.append(tool_call_result)
        print("✅ ", json.dumps(tool_call_results, ensure_ascii=False))
    return tool_call_results


if __name__ == "__main__":
    main()
