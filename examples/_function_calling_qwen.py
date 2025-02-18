# Reference: https://platform.openai.com/docs/guides/function-calling
import datetime
import json
import os

from qwen_agent.llm import get_chat_model


# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": "celsius"})
    elif "san francisco" in location.lower():
        return json.dumps(
            {"location": "San Francisco", "temperature": "72", "unit": "fahrenheit"}
        )
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": "celsius"})
    elif "深圳" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": "摄氏度"})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})


def test(fncall_prompt_type: str = "qwen"):
    llm = get_chat_model(
        {
            # Use your own model service compatible with OpenAI API:
            "model": "qwen2.5:3",
            # "model_server": "http://ollama-server:11434/v1/",  # api_base
            "model_server": "http://localhost:11434/v1/",  # api_base
            "api_key": "EMPTY",
        }
    )

    # Step 1: send the conversation and available functions to the model
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    messages = [
        {
            "role": "system",
            "content": f"你是Qwen，由阿里云创建。你是个乐于助人的助手。\n\n今天的日期为“{current_date}”，请使用中国常用的单位回答问题",
        },
        {"role": "user", "content": "深圳的天气怎么样？"},
    ]
    functions = [
        {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市和省份，例如深圳，广东",
                    },
                    "unit": {
                        "type": "string",
                        "description": "温度的单位，默认为 摄氏度",
                        "enum": ["摄氏度", "华氏度"],
                    },
                },
                "required": ["location"],
            },
        }
    ]

    print("# Assistant Response 1:")
    responses = []
    for responses in llm.chat(
        messages=messages,
        functions=functions,
        stream=True,
        # Note: extra_generate_cfg is optional
        # extra_generate_cfg=dict(
        #     # Note: if function_choice='auto', let the model decide whether to call a function or not
        #     # function_choice='auto',  # 'auto' is the default if function_choice is not set
        #     # Note: set function_choice='get_current_weather' to force the model to call this function
        #     function_choice='get_current_weather',
        # ),
    ):
        print(responses)

    # If you do not need streaming output, you can either use the following trick:
    #   *_, responses = llm.chat(messages=messages, functions=functions, stream=True)
    # or use stream=False:
    #   responses = llm.chat(messages=messages, functions=functions, stream=False)

    messages.extend(responses)  # extend conversation with assistant's reply

    # Step 2: check if the model wanted to call a function
    last_response = messages[-1]
    if last_response.get("function_call", None):

        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "get_current_weather": get_current_weather,
        }  # only one function in this example, but you can have multiple
        function_name = last_response["function_call"]["name"]
        function_to_call = available_functions[function_name]
        function_args = json.loads(last_response["function_call"]["arguments"])
        function_response = function_to_call(
            location=function_args.get("location"),
            unit=function_args.get("unit"),
        )
        print("# Function Response:")
        print(function_response)

        # Step 4: send the info for each function call and function response to the model
        messages.append(
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        )  # extend conversation with function response

        print("# Assistant Response 2:")
        for responses in llm.chat(
            messages=messages,
            functions=functions,
            stream=True,
        ):  # get a new response from the model where it can see the function response
            print(responses)


if __name__ == "__main__":
    # Run example of function calling with QwenFnCallPrompt
    test()
    while True:
        test()
