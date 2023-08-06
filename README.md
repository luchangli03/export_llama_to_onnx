# export_llama_as_onnx
Export llama as onnx files without modifying transformers modeling_llama.py

## Models to export

For llama, we will export four onnx files by the following models:

LlamaForCausalLM.lm_head

LlamaModel.embed_tokens

LlamaModel.layers

LlamaModel.norm



## Usage example

the simplest example is

```python
python export_llama.py -m model_path
```

There are also some other arguments can be used to configure the export.



## Note

Please uninstall FlashAttention before model conversion.

For kv_cache, some models use the format of [batch, head, seq, hidden], while some use [batch, seq, head, hidden]. However, the [batch, seq, head, hidden] format is much more friendly for deployment, since the memory of new cache is continuous.

The project and its developers are not responsible for the correctness of the models, and any consequences arising from the use of the exported models.

