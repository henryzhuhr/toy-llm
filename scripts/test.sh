curl -X POST http://ollama-server:11434/api/generate \
     -H "Content-Type: application/json" \
     -d '{"model": "llama3", "prompt": "tell me a joke"}'