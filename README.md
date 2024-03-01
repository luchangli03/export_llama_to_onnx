# export llama to onnx
Export LLM like llama to onnx files without modifying transformers modeling_xx_model.py

## supported models
llama (hugging face format), including llama, alpaca, etc.

Baichuan (using the llama exporter)

Alibaba Qwen 1.5/2

ChatGlm2/ChatGlm3

Gemma

bloom

## Usage example

export llama_hf
```python
# python export_llama.py -m model_dir -o out_dir --dtype fp16 # convert model to multi onnx files
python export_llama_single.py -m model_dir -o out_dir --dtype fp16 # convert model to single onnx file
```

export Qwen:
```python
python export_qwen_naive.py -m model_dir -o out_dir --dtype fp16
```
before converting Qwen, it's better to replace the rearrange ops in modeling_qwen.py to simplify the exported onnx models (please ref https://blog.csdn.net/u013701860/article/details/132123476). 

export chatglm2:
```python
python export_chatglm2.py -m model_dir --dtype fp16
```
Please use pytorch >= 2.1 (if not released, use newest nightly built version) for exporting chatglm2.
You can refer demo infer_glm2_by_onnx.py for inferring exported chatglm2 onnx

export bloom:
```python
python export_bloom_naive.py -m model_dir -o out_dir --dtype fp16
# python export_bloom.py -m model_dir -o out_dir --dtype fp16 # export more efficient and simpler model
```

Some other arguments can be used to configure the export, for example --opset can be used to set onnx opset, --add_topk_warper can be used to add topk warper to onnx model.

## Note

Please uninstall/disable FlashAttention (and maybe xformers) before model conversion.

For kv_cache, some models use the format of [batch, head, seq, hidden], while some use [batch, seq, head, hidden]. However, the [batch, seq, head, hidden] (for batch=1) or [seq, batch, head, hidden] (for both batch=1 or batch!=1) format is much more friendly for deployment, since the memory of new cache is continuous added to old cache.

To simplify the large onnx model exported by LLM, you can try https://github.com/luchangli03/onnxsim_large_model

The project (all versions) and its developers are not responsible for the correctness of the exported models, and any consequences arising from the use of the project and exported models.

