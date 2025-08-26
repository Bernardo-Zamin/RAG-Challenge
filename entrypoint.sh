#!/bin/bash

# Inicia o servidor Ollama em background
ollama serve &

# Aguarda ele subir (ajuste conforme necessário)
sleep 5

# Puxa o modelo desejado
ollama pull tinyllama

# Aguarda o processo do Ollama
wait
