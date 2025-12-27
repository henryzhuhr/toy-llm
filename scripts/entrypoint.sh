#!/bin/bash

# Start Ollama in the background.
/bin/ollama serve &
# Record Process ID.
pid=$!

# Pause for Ollama to start.
sleep 5


echo "🟡 Get value OLLAMA_MODELS=$OLLAMA_MODELS"

OLLAMA_MODELS=${OLLAMA_MODELS:-qwen2.5:0.5b}

echo "🔴 Retrieve $OLLAMA_MODELS model..."
ollama run $OLLAMA_MODELS --verbose
echo "🟢 Done!"

# Wait for Ollama process to finish.
wait $pid