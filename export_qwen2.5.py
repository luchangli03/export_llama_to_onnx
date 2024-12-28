import os
import logging
import argparse
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


class QwenForCausalLMWrapper(nn.Module):
    def __init__(self, model, config, args):
        super().__init__()
        self.model = model
        self.config = config
        self.args = args
        self.layer_num = len(model.model.layers)

    def forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        key_cache,
        value_cache,
        cache_position,
    ):

        use_cache = True
        output_attentions = False
        output_hidden_states = False
        return_dict = True
        num_logits_to_keep = 1

        past_key_values = DynamicCache(self.layer_num)
        past_key_values.key_cache = key_cache
        past_key_values.value_cache = value_cache
        past_key_values._seen_tokens = int(cache_position)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=None,
            labels=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
        )

        logits = outputs.logits

        key_cache_out = [tensor for tensor in outputs.past_key_values.key_cache]
        value_cache_out = [tensor for tensor in outputs.past_key_values.value_cache]

        topk_indices = None
        if self.args.add_topk_warper > 0:
            logging.warning("add topk to glm model")
            if self.args.topk < 0:
                raise ValueError("topk {} is invalid")
            if self.args.topk > 1:
                values, topk_indices = torch.topk(logits, k=self.args.topk, dim=-1)
            else:
                topk_indices = torch.argmax(logits, dim=-1)

        topk_indices = [topk_indices] if topk_indices is not None else []
        outputs = [logits] + key_cache_out + value_cache_out + topk_indices
        return outputs


def export_qwen_to_single_onnx(model, config, dtype, args, model_name):
    qwen_model_wrapper = QwenForCausalLMWrapper(model, config, args)
    onnx_file_name = os.path.join(args.out_dir, f"{model_name}.onnx")
    layer_num = len(model.model.layers)

    hidden_size = config.hidden_size
    head_num = config.num_attention_heads
    head_dim = hidden_size // head_num
    num_key_value_heads = config.num_key_value_heads

    batch = 1
    N = 1
    sumN = 38
    lastSum = sumN - N

    input_ids = torch.ones([batch, N], dtype=torch.int64).to(args.device)
    attention_mask = torch.ones([batch, sumN], dtype=torch.int64).to(args.device)
    # attention_mask = torch.ones([1, 1, N, sumN], dtype=dtype).to(args.device)
    position_ids = torch.Tensor([lastSum]).to(torch.int64).to("cuda").reshape(batch, N)
    cache_position = torch.Tensor([lastSum]).to(torch.int64).to("cuda")

    in_names = ["input_ids", "attention_mask", "position_ids"]

    dynamic_axes = {
        'input_ids': {1: 'N', },
        'attention_mask': {1: "sumN"},
        # 'attention_mask': {2: "N", 3: "sumN"},
        "position_ids": {1: 'N', },
    }

    kv_caches_in = []
    out_names = ["lm_logits"]

    kv_cache_in_shape = [batch, num_key_value_heads, lastSum, head_dim]
    kv_cache_in_dyn_axes = {2: "sumN-N"}

    print("kv_cache_in_shape:", kv_cache_in_shape)

    key_cache = []
    value_cache = []

    key_cache_names_in = []
    value_cache_names_in = []
    key_cache_names_out = []
    value_cache_names_out = []

    for i in range(layer_num):
        past_key_in = torch.randn(kv_cache_in_shape, dtype=dtype).to(args.device)
        past_value_in = torch.randn(kv_cache_in_shape, dtype=dtype).to(args.device)

        key_cache.append(past_key_in)
        value_cache.append(past_value_in)

        key_cache_names_in.append(f"past_key_in{i}")
        value_cache_names_in.append(f"past_value_in{i}")
        key_cache_names_out.append(f"past_key{i}")
        value_cache_names_out.append(f"past_value{i}")

        dynamic_axes[f"past_key_in{i}"] = kv_cache_in_dyn_axes
        dynamic_axes[f"past_value_in{i}"] = kv_cache_in_dyn_axes

    input_datas = (input_ids, attention_mask, position_ids, key_cache, value_cache, cache_position)

    in_names.extend(key_cache_names_in)
    in_names.extend(value_cache_names_in)
    in_names.append("cache_position")

    out_names.extend(key_cache_names_out)
    out_names.extend(value_cache_names_out)
    if args.add_topk_warper > 0:
        out_names.append("topk_indices")

    # results = qwen_model_wrapper(*input_datas)
    print("infer finish")
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
    suitable transformers: 4.47.1
    you can modify Qwen2Model.forward to directly use expanded 4D mask to get a simpler onnx:
    '''
    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )
    '''
    if len(attention_mask.shape) ==2:
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
    else:
        causal_mask = attention_mask
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='export qwen',
    )
    parser.add_argument('-m', '--model_path', required=True, type=str)
    parser.add_argument('-o', '--out_dir', required=False, type=str, default="")
    parser.add_argument('--opset', required=False, type=int, default=17)
    parser.add_argument('-d', '--device', required=False, type=str, choices=["cpu", "cuda"], default="cuda")
    parser.add_argument('-p', '--dtype', required=False, type=str,
                        choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument('--add_topk_warper', action='store_true')
    parser.add_argument('--topk', required=False, type=int, default=4)
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    logging.warning(f"*** Note: please apply modications to transformers model before conversion: {model_modication_note}")

    export_qwen(args)
