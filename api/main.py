import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()


@app.post("/v1/chat/completions")
async def chat_completions(request: dict):
    if request.get("stream"):
        return StreamingResponse(
            generate_stream(request), media_type="text/event-stream"
        )
    else:
        # 返回非流式响应
        return {
            "choices": [{"message": {"role": "assistant", "content": "你的AI回复"}}]
        }


async def generate_stream(request):
    content = "你的AI流式回复"
    for token in content:
        chunk = {"choices": [{"delta": {"role": "assistant", "content": token}}]}
        yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"


@app.get("/api/models")
async def get_models():
    return {
        "data": [
            {
                "id": "toy-llm-model",
                "object": "model",
                "created": 1697049600,
                "owned_by": "toy-llm",
                "permission": [],
            }
        ]
    }


# curl -H "Authorization: Bearer YOUR_API_KEY" http://localhost:8000/api/models
# curl -H "Authorization: Bearer YOUR_API_KEY" http://toy-llm:8000/api/models
def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
