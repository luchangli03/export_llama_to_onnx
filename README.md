# export_llama_as_onnx
export llama as onnx

Models to export:

LlamaForCausalLM.lm_head

LlamaModel.embed_tokens

LlamaModel.layers

LlamaModel.norm



Usage example:

```python
python export_llama.py -m model_path
```

