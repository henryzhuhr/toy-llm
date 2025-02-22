| 字段        | 描述                                                           |
| ----------- | -------------------------------------------------------------- |
| name        | 函数的名称 (例如 get_weather)                                  |
| description | 关于何时以及如何使用该功能的详细信息                           |
| parameters  | 使用 [JSON schema](https://json-schema.org) 定义函数的输入参数 |

```json
{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "检索给定位置的当前天气。",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "城市和省区市，例如：深圳, 广东"
                },
                "units": {
                    "type": "string",
                    "enum": [
                        "摄氏度",
                        "华氏度"
                    ],
                    "description": "单位，温度将返回的。"
                }
            },
            "required": [
                "location",
                "units"
            ],
            "additionalProperties": false
        },
        "strict": true
    }
}
```

Best practices for defining functions

Write clear and detailed function names, parameter descriptions, and instructions.

Explicitly describe the purpose of the function and each parameter (and its format), and what the output represents.

Use the system prompt to describe when (and when not) to use each function. Generally, tell the model exactly what to do.

Include examples and edge cases, especially to rectify any recurring failures. (Note: Adding examples may hurt performance for reasoning models.)