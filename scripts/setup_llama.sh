#!/bin/bash
# Setup LLaMA-3.2-3B after ModelScope download completes.
# Usage: bash scripts/setup_llama.sh

MS_DIR="/root/autodl-tmp/models/Llama-3.2-3B-ms-cache/LLM-Research/Llama-3___2-3B"
TARGET="/root/autodl-tmp/models/Llama-3.2-3B"

# Check download completion
if [ ! -f "$MS_DIR/model-00001-of-00002.safetensors" ] || [ ! -f "$MS_DIR/model-00002-of-00002.safetensors" ]; then
    echo "ERROR: ModelScope download not yet complete."
    echo "Check: ls -la $MS_DIR/*.safetensors"
    exit 1
fi

# Replace corrupt HF shards with ModelScope versions
echo "Copying safetensors from ModelScope cache..."
cp "$MS_DIR/model-00001-of-00002.safetensors" "$TARGET/model-00001-of-00002.safetensors"
cp "$MS_DIR/model-00002-of-00002.safetensors" "$TARGET/model-00002-of-00002.safetensors"
cp "$MS_DIR/model.safetensors.index.json" "$TARGET/model.safetensors.index.json"

echo "Verifying model load..."
export CUDA_VISIBLE_DEVICES=0
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
t = AutoTokenizer.from_pretrained('$TARGET')
m = AutoModelForCausalLM.from_pretrained('$TARGET', torch_dtype=torch.bfloat16, device_map='auto')
print(f'Model loaded: {m.config.architectures}, params={sum(p.numel() for p in m.parameters())/1e9:.2f}B')
print(f'Tokenizer: pad={t.pad_token} eos={t.eos_token}')
# Quick generation test
msgs = [{'role': 'system', 'content': 'You are a FOL expert.'}, {'role': 'user', 'content': 'Hello'}]
text = t.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
inputs = t([text], return_tensors='pt').to(m.device)
out = m.generate(**inputs, max_new_tokens=20, pad_token_id=t.pad_token_id)
print(f'Generation test: {t.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)[:100]}')
print('SUCCESS: Model ready for GRPO training.')
"
