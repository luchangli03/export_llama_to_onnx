import os
import logging
import argparse
import torch
from torch import nn
from transformers import AutoTokenizer, BloomForCausalLM
import math

DTYPE_NP_2_TORCH = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class BloomForCausalLMWrapper(nn.Module):
    def __init__(self, model, config, args):
        super().__init__()
        self.model = model
        self.config = config
        self.layer_num = config.n_layer

        self.args = args
        self.max_seq_len = self.config.seq_length

    def forward(self, input_ids, position_ids, attention_mask, kv_caches):

        past_key_values = []
        for i in range(self.layer_num):
            past_key_values.append((kv_caches[2 * i], kv_caches[2 * i + 1]))

        inputs_embeds = self.model.transformer.word_embeddings(input_ids)
        hidden_states = self.model.transformer.word_embeddings_layernorm(inputs_embeds)

        head_mask = self.model.transformer.get_head_mask(None, self.config.n_layer)

        # alibi = self.model.transformer.build_alibi_tensor(
        #     attention_mask, self.model.transformer.num_heads, dtype=hidden_states.dtype)

        alibi = build_alibi_tensor1(
            position_ids, self.model.transformer.num_heads, dtype=hidden_states.dtype)

        batch_size, seq_length = input_ids.shape

        past_key_values_length = past_key_values[0][0].shape[2]

        decoder_layers = self.model.transformer.h

        kv_caches_out = []

        for i, (block, layer_past) in enumerate(zip(decoder_layers, past_key_values)):

            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                use_cache=True,
                output_attentions=False,
                alibi=alibi,
            )
            hidden_states = outputs[0]

            kv_caches_out.extend(outputs[1],)

        hidden_states = hidden_states[:, -1:, :]

        hidden_states = self.model.transformer.ln_f(hidden_states)

        lm_logits = self.model.lm_head(hidden_states)

        topk_outputs = []
        if self.args.add_topk_warper:
            logging.warning("add topk to model")
            if self.args.topk < 0:
                raise ValueError("topk {} is invalid")
            topk_outputs = torch.topk(lm_logits, k=self.args.topk, dim=-1)

        return lm_logits, *kv_caches_out, *topk_outputs


def build_alibi_tensor1(position_ids: torch.Tensor, num_heads: int, dtype: torch.dtype) -> torch.Tensor:
    """
    using position_ids but not attention_mask to simplify abili computation
    the exported onnx graph will also be much more simplier
    """
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), device=position_ids.device, dtype=torch.float32
    )
    powers = torch.arange(1, 1 + closest_power_of_2, device=position_ids.device, dtype=torch.int32)
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != num_heads:
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))), device=position_ids.device, dtype=torch.float32
        )
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = torch.arange(1, 1 + 2 * num_remaining_heads, 2, device=position_ids.device, dtype=torch.int32)
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)

    batch_size, seq_length = position_ids.shape

    position_ids = position_ids.reshape(batch_size, 1, seq_length)

    alibi = slopes[..., None] * position_ids
    return alibi.reshape(batch_size * num_heads, 1, seq_length).to(dtype)


def export_bloom_model(model, config, dtype, args, model_name):
    onnx_file_name = os.path.join(args.out_dir, f"{model_name}.onnx")
    model_wrapper = BloomForCausalLMWrapper(model, config, args)

    hidden_size = config.hidden_size

    batch = 1
    N = 1
    sumN = 32
    lastSum = sumN - N

    layer_num = config.n_layer

    input_ids = torch.ones([batch, N], dtype=torch.int64).to(args.device)
    # directly create expanded attention_mask to avoid _prepare_attn_mask in model and export this to onnx
    # thus can simplify onnx and improve performance
    attention_mask = torch.ones([batch, 1, N, sumN], dtype=torch.bool).to(args.device)

    # use position ids to simplify position embeding computing
    position_ids = torch.arange(start=0, end=N, dtype=torch.int64).to(args.device)
    position_ids = position_ids.reshape(1, N)

    in_names = ["input_ids", "position_ids", "attention_mask"]
    input_datas = [input_ids, position_ids, attention_mask]

    dynamic_axes = {
        'input_ids': {1: 'N', },
        'position_ids': {1: 'N', },
        'attention_mask': {2: "N", 3: "sumN"},
    }

    kv_caches_in = []
    out_names = ["lm_logits"]

    n_head = config.n_head
    cache_channel = hidden_size // n_head

    k_cache_in_shape = [n_head, cache_channel, lastSum]
    v_cache_in_shape = [n_head, lastSum, cache_channel]
    k_cache_dyn_axes = {2: "lastSum"}
    v_cache_dyn_axes = {1: "lastSum"}

    for i in range(layer_num):
        past_key_in = torch.randn(k_cache_in_shape, dtype=dtype).to(args.device)
        past_value_in = torch.randn(v_cache_in_shape, dtype=dtype).to(args.device)

        kv_caches_in.extend([past_key_in, past_value_in])
        in_names.extend([f"past_key_in{i}", f"past_value_in{i}"])
        out_names.extend([f"past_key{i}", f"past_value{i}"])

        dynamic_axes[f"past_key_in{i}"] = k_cache_dyn_axes
        dynamic_axes[f"past_value_in{i}"] = v_cache_dyn_axes

    input_datas.append(kv_caches_in)

    if args.add_topk_warper > 0:
        out_names.extend(["logits_topk_value", "logits_topk_idx"])

    torch.onnx.export(
        model_wrapper,
        tuple(input_datas),
        onnx_file_name,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=in_names,
        output_names=out_names,
        dynamic_axes=dynamic_axes,
    )


def export_bloom(args):
    device = args.device
    dtype = DTYPE_NP_2_TORCH[args.dtype]

    print(f"begin load model from {args.model_path}")
    # tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = BloomForCausalLM.from_pretrained(args.model_path, trust_remote_code=True).half()
    if args.dtype == "float32":
        model.float()
        print("convert model to float")

    if args.device == "cuda":
        model.cuda()
        print("convert model to cuda")

    model = model.eval()

    print(f"finish load model from {args.model_path}")
    config = model.config

    print("begin export model")
    export_bloom_model(model, config, dtype, args, "bloom_model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='export bloom',
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

    export_bloom(args)
