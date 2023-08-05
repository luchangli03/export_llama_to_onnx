# export_llama_as_onnx
Export llama as onnx files without modifying modeling_llama.py

## Models to export

LlamaForCausalLM.lm_head

LlamaModel.embed_tokens

LlamaModel.layers

LlamaModel.norm



## Usage example

```python
python export_llama.py -m model_path
```

