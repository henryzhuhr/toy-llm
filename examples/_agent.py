from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI

llm_cfg = {
    "model": "qwen:7b",
    "model_server": "http://ollama-server:11434/api",  # base_url, also known as api_base
    "api_key": "EMPTY",
}


# /api/chat/completions
def app_gui():
    # Define the agent
    bot = Assistant(
        llm=llm_cfg,
        name="Assistant",
        description="使用RAG检索并回答，支持文件类型：PDF/Word/PPT/TXT/HTML。",
    )
    chatbot_config = {
        "prompt.suggestions": [
            {"text": "如何办理保险？"},
            {"text": "如何理赔？"},
        ]
    }
    WebUI(bot, chatbot_config=chatbot_config).run()


if __name__ == "__main__":
    app_gui()
