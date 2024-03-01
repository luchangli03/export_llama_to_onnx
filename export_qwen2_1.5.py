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
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        logits = outputs.logits
 
        kv_caches_out = []
        for past_kv in outputs.past_key_values:
            kv_caches_out.extend(past_kv)
 
        topk_outputs = []
        if self.args.add_topk_warper > 0:
            logging.warning("add topk to glm model")
            if self.args.topk < 0:
                raise ValueError("topk {} is invalid")
            topk_outputs = torch.topk(logits, k=self.args.topk, dim=-1)
        return logits, *kv_caches_out, *topk_outputs
 
 
def export_qwen_to_single_onnx(model, config, dtype, args, model_name):
    qwen_model_wrapper = QwenForCausalLMWrapper(model, config, args)
    onnx_file_name = os.path.join(args.out_dir, f"{model_name}.onnx")
    layer_num = len(model.model.layers)
 
    hidden_size = config.hidden_size
    head_num = config.num_attention_heads
    head_dim = hidden_size // head_num
 
    batch = 1
    N = 1
    sumN = 32
    lastSum = sumN - N
 
    input_ids = torch.ones([batch, N], dtype=torch.int64).to(args.device)
    # attention_mask = torch.ones([batch, sumN], dtype=torch.int64).to(args.device)
    attention_mask = torch.zeros([1, 1, N, sumN], dtype=dtype).to(args.device)
    position_ids = torch.ones([batch, N], dtype=torch.int64).to(args.device)
 
    in_names = ["input_ids", "attention_mask", "position_ids"]
 
    dynamic_axes = {
        'input_ids': {1: 'N', },
        'attention_mask': {2: "N", 3: "sumN"},
        "position_ids": {1: 'N', },
    }
 
    kv_caches_in = []
    out_names = ["lm_logits"]
 
    kv_cache_in_shape = [batch, head_num, lastSum, head_dim]
    kv_cache_dyn_axes = {2: "sumN-N"}
 
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
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]
 
    print(f"begin load model from {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, device_map=device, trust_remote_code=True, torch_dtype=dtype).eval()
 
    # model.model.layers = model.model.layers[:1]  # debug
 
    print(f"finish load model from {args.model_path}")
    config = model.config
    print("config:", config)
 
    print(f"begin export qwen")
    export_qwen_to_single_onnx(model, config, dtype, args, "qwen_onnx")
 
 
model_modication_note = """
modication 1: in Qwen2ForCausalLM.forward
        hidden_states = outputs[0]
        hidden_states = hidden_states[:,-1:,:] # <<--
        logits = self.lm_head(hidden_states)
modication 2: in Qwen2Model.forward
        '''
        if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Qwen2. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )
        if self._attn_implementation == "flash_attention_2":
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._attn_implementation == "sdpa" and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=self.config.sliding_window,
            )
        '''
modication 3: in Qwen2RotaryEmbedding.forward
        return (self.cos_cached.to(dtype=x.dtype), self.sin_cached.to(dtype=x.dtype))
        # return (
        #     self.cos_cached[:seq_len].to(dtype=x.dtype),
        #     self.sin_cached[:seq_len].to(dtype=x.dtype),
        # )
"""
 
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
 
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
 
    logging.warning(f"*** Note: please apply modications to model before conversion: {model_modication_note}")
 
    export_qwen(args)
