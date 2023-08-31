# export llama to onnx
Export llama to onnx files without modifying transformers modeling_llama.py

## support export llama_hf (alpaca, etc.), Alibaba Qwen by export_llama.py

## support export ChatGlm2 by export_chatglm2.py
Please use pytorch 2.1 (if not released, use newest nightly built version) for exporting chatglm2.
You can refer demo infer_glm2_by_onnx.py for inferring exported chatglm2 onnx

## support export bloom by export_bloom.py

## Models to export

For llama, we will export four onnx files by the following models:

LlamaForCausalLM.lm_head

LlamaModel.embed_tokens

LlamaModel.layers

LlamaModel.norm

Actually it's very easy to convert all these sub models in a single onnx model, we show this in export chatglm2.py, export_llama_single.py.


## Usage example

convert llama_hf
```python
python export_llama.py -m model_dir --dtype fp16 # convert model to multi onnx files
# python export_llama_single.py -m model_dir --dtype fp16 # convert model to single onnx file
```

convert Qwen:
```python
python export_qwen_naive.py -m model_dir -o out_dir
```
before converting Qwen, it's better to replace the rearrange ops in modeling_qwen.py to simplify the exported onnx models (please ref https://blog.csdn.net/u013701860/article/details/132123476). 

convert chatglm2:
```python
python export_chatglm2.py -m model_dir --dtype fp16 # [--add_topk_warper 1]
```

Some other arguments can be used to configure the export, such as the opset, output dirs.



## Note

Please uninstall/disable FlashAttention (and maybe xformers) before model conversion.

For kv_cache, some models use the format of [batch, head, seq, hidden], while some use [batch, seq, head, hidden]. However, the [batch, seq, head, hidden] format is much more friendly for deployment, since the memory of new cache is continuous.

The project (all versions) and its developers are not responsible for the correctness of the exported models, and any consequences arising from the use of the project and exported models.

