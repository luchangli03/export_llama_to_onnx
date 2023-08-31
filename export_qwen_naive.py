import os
import logging
import argparse
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
 
 
class QwenForCausalLMWrapper(nn.Module):
    def __init__(self, model, config, args):
        super().__init__()
        self.model = model
        self.config = config
        self.args = args
 
    def forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        past_key_values,
    ):
        transformer_outputs = self.model.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=None,
            position_ids=position_ids,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=None,
        )
 
        kv_caches_out = []
        for past_kv in transformer_outputs.past_key_values:
            kv_caches_out.extend(past_kv)
 
        hidden_states = transformer_outputs[0]
 
        hidden_states = hidden_states[:, -1, :]
        lm_logits = self.model.lm_head(hidden_states)
 
        topk_outputs = []
        if self.args.add_topk_warper > 0:
            logging.warning("add topk to glm model")
            if self.args.topk < 0:
                raise ValueError("topk {} is invalid")
            topk_outputs = torch.topk(lm_logits, k=self.args.topk, dim=-1)
 
        return lm_logits, *kv_caches_out, *topk_outputs
 
 
def export_qwen_to_single_onnx(model, config, dtype, args, model_name):
    qwen_model_wrapper = QwenForCausalLMWrapper(model, config, args)
 
    onnx_file_name = os.path.join(args.out_dir, f"{model_name}.onnx")
 
    layer_num = config.num_hidden_layers
 
    hidden_size = config.hidden_size
    head_num = config.num_attention_heads
    head_dim = hidden_size // head_num
 
    batch = 1
    N = 1
    sumN = 32
    lastSum = sumN - N
 
    input_ids = torch.ones([batch, N], dtype=torch.int64).to(args.device)
    attention_mask = torch.ones([batch, sumN], dtype=torch.int64).to(args.device)
    position_ids = torch.ones([batch, N], dtype=torch.int64).to(args.device)
 
    in_names = ["input_ids", "attention_mask", "position_ids"]
 
    dynamic_axes = {
        'input_ids': {0: 'batch', 1: 'N', },
        'attention_mask': {0: 'batch', 1: "sumN"},
        "position_ids": {0: 'batch', 1: 'N', },
    }
 
    kv_caches_in = []
    out_names = ["lm_logits"]
 
    kv_cache_in_shape = [batch, lastSum, head_num, head_dim]
    kv_cache_dyn_axes = {0: 'batch', 1: "lastSum"}
 
    past_key_values = []
 
    for i in range(layer_num):
        past_key_in = torch.randn(kv_cache_in_shape, dtype=dtype).to(args.device)
        past_value_in = torch.randn(kv_cache_in_shape, dtype=dtype).to(args.device)
 
        kv_caches_in.extend([past_key_in, past_value_in])
        in_names.extend([f"past_key_in{i}", f"past_value_in{i}"])
        out_names.extend([f"past_key{i}", f"past_value{i}"])
 
        dynamic_axes[f"past_key_in{i}"] = kv_cache_dyn_axes
        dynamic_axes[f"past_value_in{i}"] = kv_cache_dyn_axes
 
        past_key_values.append((past_key_in, past_value_in))
 
    input_datas = (input_ids, attention_mask, position_ids, past_key_values)
 
    torch.onnx.export(
        qwen_model_wrapper,
        input_datas,
        onnx_file_name,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=in_names,
        output_names=out_names,
        dynamic_axes=dynamic_axes,
    )
 
 
def export_qwen(args):
    device = args.device
 
    dtype_cfg = {
        "fp32": False,
        "fp16": False,
        "bf16": False,
    }
 
    if args.dtype == "float32":
        dtype_cfg["fp32"] = True
        dtype = torch.float32
    elif args.dtype == "float16":
        dtype_cfg["fp16"] = True
        dtype = torch.float16
    elif args.dtype == "bfloat16":
        dtype_cfg["bf16"] = True
        dtype = torch.bfloat16
 
    print(f"begin load model from {args.model_path}")
    # tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
 
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, device_map=device, trust_remote_code=True, **dtype_cfg).eval()
 
    print(f"finish load model from {args.model_path}")
    config = model.config
    print("config:", config)
 
    print(f"begin export qwen")
    export_qwen_to_single_onnx(model, config, dtype, args, "qwen_onnx")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='export qwen',
    )
    parser.add_argument('-m', '--model_path', required=True, type=str)
    parser.add_argument('-o', '--out_dir', required=False, type=str, default="")
    parser.add_argument('--opset', required=False, type=int, default=15)
    parser.add_argument('-d', '--device', required=False, type=str, choices=["cpu", "cuda"], default="cuda")
    parser.add_argument('-p', '--dtype', required=False, type=str,
                        choices=["float32", "float16", "bfloat16"], default="float16")
 
    parser.add_argument('--add_topk_warper', action='store_true')
    parser.add_argument('--topk', required=False, type=int, default=4)
 
    args = parser.parse_args()
 
    if args.dtype not in ["float32", "float16", "bfloat16"]:
        raise ValueError("dtype is invalid")
 
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
 
    export_qwen(args)
 
    logging.warning(
        """
        you can optimize exported onnx by follwing methods:
        1. replace einops rearrange in modeling_qwen.py
        2. replace the attention_mask by expanded attention_mask to avoid expanding mask in onnx
        3. optimize RotaryEmbedding to compute position embeding of max seq len,
        and using tensor gather to get embeding of each iteration, but not recompute it, just like llama and chatglm2
        """
    )
