import json
from ollama import Client

from datetime import datetime

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_temperature",
            "description": "获取某个位置的当前温度。",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": '获取温度的位置，格式为"城市、州、国家"。',
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["摄氏度", "华氏温度"],
                        "description": '返回温度的单位。默认为"摄氏度"。',
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_temperature_date",
            "description": "获取位置和日期的温度。",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "获取温度的位置，格式为“城市、州、国家”。",
                    },
                    "date": {
                        "type": "string",
                        "description": "获取温度的日期，格式为“年-月-日”。",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["摄氏度", "华氏温度"],
                        "description": "返回温度的单位。默认为“摄氏度”。",
                    },
                },
                "required": ["location", "date"],
            },
        },
    },
]
TOOLS_EN = [
    {
        "type": "function",
        "function": {
            "name": "get_current_temperature",
            "description": "Get current temperature at a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": 'The location to get the temperature for, in the format "City, State, Country".',
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": 'The unit to return the temperature in. Defaults to "celsius".',
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_temperature_date",
            "description": "Get temperature at a location and date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": 'The location to get the temperature for, in the format "City, State, Country".',
                    },
                    "date": {
                        "type": "string",
                        "description": 'The date to get the temperature for, in the format "Year-Month-Day".',
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": 'The unit to return the temperature in. Defaults to "celsius".',
                    },
                },
                "required": ["location", "date"],
            },
        },
    },
]


def get_current_temperature(location: str, unit: str = "celsius"):
    """Get current temperature at a location.

    Args:
        location: The location to get the temperature for, in the format "City, State, Country".
        unit: The unit to return the temperature in. Defaults to "celsius". (choices: ["celsius", "fahrenheit"])

    Returns:
        the temperature, the location, and the unit in a dict
    """
    return {
        "temperature": 26.1,
        "location": location,
        "unit": unit,
    }


def get_temperature_date(location: str, date: str, unit: str = "celsius"):
    """Get temperature at a location and date.

    Args:
        location: The location to get the temperature for, in the format "City, State, Country".
        date: The date to get the temperature for, in the format "Year-Month-Day".
        unit: The unit to return the temperature in. Defaults to "celsius". (choices: ["celsius", "fahrenheit"])

    Returns:
        the temperature, the location, the date and the unit in a dict
    """
    return {
        "temperature": 25.9,
        "location": location,
        "date": date,
        "unit": unit,
    }


def get_function_by_name(name):
    if name == "get_current_temperature":
        return get_current_temperature
    if name == "get_temperature_date":
        return get_temperature_date


def main():

    current_date = datetime.now().strftime("%Y-%m-%d")
    MESSAGES = [
        {
            "role": "system",
            "content": f"你是Qwen，由阿里云创建。你是个乐于助人的助手。\n\n今天的日期为“{current_date}”，请使用中国常用的单位回答问题",
        },
        {
            "role": "user",
            "content": "深圳现在的气温是多少？明天怎么样？",
        },
    ]

    tools = TOOLS_EN
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
        for tool_call in tool_calls:
            if fn_call := tool_call.get("function"):
                fn_name: str = fn_call["name"]
                fn_args: dict = fn_call["arguments"]

                print("\n❓ ", end="")
                print(f"function: {fn_name}")
                print(f"arguments:")
                for k, v in fn_args.items():
                    print(f"\t{k}: {v}")

                fn_res: str = json.dumps(get_function_by_name(fn_name)(**fn_args))

                tool_call_result = {
                    "role": "tool",
                    "name": fn_name,
                    "content": fn_res,
                }
                messages.append(tool_call_result)

                print("🤖 ", tool_call_result)
                for key, value in tool_call_result.items():
                    if key == "content":
                        _val = json.loads(value)
                        print("\tcontent:")
                        for k, v in _val.items():
                            print(f"\t\t{k}: {v}")
                    else:
                        print(f"\t{key}: {value}")


if __name__ == "__main__":
    main()
