import os
import argparse
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

 
class LLMForCausalLMWrapper(nn.Module):
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
        output_attentions=False,
        output_hidden_states=False,
        use_cache=True,
    ):
        """
        Note: you can modify modeling_gemma.py to make the converted model more efficient:
        hidden_states = outputs[0]
        hidden_states = hidden_states[:,-1:,:]
        logits = self.lm_head(hidden_states)
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=None,
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
 
 
def export_llm_to_single_onnx(model, config, dtype, args, model_name):
    llama_model_wrapper = LLMForCausalLMWrapper(model, config, args)
 
    onnx_file_name = os.path.join(args.out_dir, f"{model_name}.onnx")
 
    layer_num = len(model.model.layers)
 
    hidden_size = config.hidden_size
    kv_head_num = config.num_key_value_heads
    head_dim = config.head_dim
 
    batch = 1
    N = 1
    sumN = 32
    lastSum = sumN - N
 
    input_ids_shape = [batch, N]
    input_ids = torch.ones(input_ids_shape, dtype=torch.int64).to(args.device)
    # Note: orig atten_mask shape is [1, sumN]
    attention_mask = torch.randn([batch, 1, N, sumN], dtype=dtype).to(args.device)
    position_ids = torch.ones([batch, N], dtype=torch.int64).to(args.device)
 
    in_names = ["input_ids", "attention_mask", "position_ids"]
 
    dynamic_axes = {
        'input_ids': {1: 'N', },
        'attention_mask': {2: 'N', 3: 'sumN'},
        "position_ids": {1: 'N', },
    }
    if args.dyn_batch:
        dynamic_axes['input_ids'][0] = "batch"
        dynamic_axes['attention_mask'][0] = "batch"
        dynamic_axes['position_ids'][0] = "batch"
 
    kv_caches_in = []
    out_names = ["lm_logits"]
 
    kv_cache_in_shape = [1, kv_head_num, lastSum, head_dim]
    kv_cache_dyn_axes = {2: "sumN-N"}
 
    if args.dyn_batch:
        kv_cache_dyn_axes[0] = "batch"
 
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
        llama_model_wrapper,
        input_datas,
        onnx_file_name,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=in_names,
        output_names=out_names,
        dynamic_axes=dynamic_axes,
    )
 
 
def export_llama(args):
    device = args.device
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]
 
    print(f"begin load model from {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, device_map=device, torch_dtype=dtype, trust_remote_code=True).eval()
 
    # model.model.layers = model.model.layers[:1]  # only export one layer for debug
 
    print(f"finish load model from {args.model_path}")
    config = model.config
    print("config:", config)
 
    print(f"begin export llm")
    export_llm_to_single_onnx(model, config, dtype, args, "llm_onnx")


model_modication_note = """
modication 1:
        hidden_states = outputs[0]
        hidden_states = hidden_states[:,-1:,:] # <<--
        logits = self.lm_head(hidden_states)
modication 2:
class GemmaModel(GemmaPreTrainedModel):
    def forward(
        # causal_mask = self._update_causal_mask(attention_mask, inputs_embeds)
        causal_mask = attention_mask
 
class GemmaSdpaAttention(GemmaAttention):
    def forward(
        # if attention_mask is not None and cache_position is not None:
        #     causal_mask = causal_mask[:, :, cache_position, : key_states.shape[-2]]
"""
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='export llm',
    )
    parser.add_argument('-m', '--model_path', required=True, type=str)
    parser.add_argument('-o', '--out_dir', required=False, type=str, default="")
    parser.add_argument('--opset', required=False, type=int, default=15)
    parser.add_argument('-d', '--device', required=False, type=str, choices=["cpu", "cuda"], default="cuda")
    parser.add_argument('-p', '--dtype', required=False, type=str,
                        choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument('--add_topk_warper', required=False, type=int, default=0)
    parser.add_argument('--topk', required=False, type=int, default=4)
    parser.add_argument('--dyn_batch', action='store_true')
 
    args = parser.parse_args()
  
    logging.warning(f"*** Note: please apply modications to model before conversion: {model_modication_note}")

    export_llama(args)
