import json
from ollama import Client

from datetime import datetime


TOOLS = []


def get_function_by_name(name):
    def func(**kwargs):
        return ""

    func_map = {
        # "getIncidentAgg": func,
        "getIncidentList": func,
    }

    func_handler = func_map.get(name, lambda: {"error": "function not found"})
    print(f"🟢 function: {name}")

    return func_handler


def main():

    current_date = datetime.now().strftime("%Y-%m-%d")
    MESSAGES = [
        # {
        #     "role": "system",
        #     "content": f"今天的日期为“{current_date}”",
        # },
        {
            "role": "user",
            "content": "显示所有标记为特别关注的事件。",
        },
    ]

    tools = TOOLS
    messages = MESSAGES[:]

    model_name = "qwen2.5:3b"

    client = Client(
        # host="http://ollama-server:11434",
        host="http://localhost:11434",
        headers={"x-some-header": "some-value"},
    )
    response = client.chat(
        model=model_name,
        messages=messages,
        tools=tools,  # 输入工具，会构造成工具调用
    )
    print("LLM response:", response)
    print()
    messages.append(response["message"])
    print(response["message"])

    if tool_calls := messages[-1].get("tool_calls", None):
        # print("\n🔵 ", tool_calls)
        tool_call_results = []

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

        # for tool_call in tool_calls:
        #     if fn_call := tool_call.get("function"):
        #         fn_name: str = fn_call["name"]
        #         fn_args: dict = fn_call["arguments"]

        #         print("\n❓ ", end="")
        #         print(f"function: {fn_name}")
        #         print(f"arguments:")
        #         for k, v in fn_args.items():
        #             print(f"\t{k}: {v}")

        #         fn_res: str = json.dumps(get_function_by_name(fn_name)(**fn_args))

        #         tool_call_result = {
        #             "role": "tool",
        #             "name": fn_name,
        #             "content": fn_res,
        #         }
        #         messages.append(tool_call_result)


if __name__ == "__main__":
    main()
