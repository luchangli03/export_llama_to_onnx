import os
import logging
import argparse
import torch
from torch import nn
from transformers import AutoTokenizer, BloomForCausalLM


class BloomForCausalLMWrapper(nn.Module):
    def __init__(self, model, config, args):
        super().__init__()
        self.model = model
        self.config = config
        self.layer_num = config.n_layer

        self.args = args

    def forward(self, input_ids, attention_mask, kv_caches):

        past_key_values = []
        for i in range(self.layer_num):
            past_key_values.append((kv_caches[2 * i], kv_caches[2 * i + 1]))

        transformer_outputs = self.model.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=None,
            use_cache=True,
        )

        hidden_states = transformer_outputs.last_hidden_state
        hidden_states = hidden_states[:, -1:, :]
        lm_logits = self.model.lm_head(hidden_states)

        past_key_values = transformer_outputs.past_key_values
        kv_caches_out = []
        for layer_cache in past_key_values:
            kv_caches_out.extend(list(layer_cache))

        topk_outputs = []
        if self.args.add_topk_warper:
            logging.warning("add topk to model")
            if self.args.topk < 0:
                raise ValueError("topk {} is invalid")
            topk_outputs = torch.topk(lm_logits, k=self.args.topk, dim=-1)

        return lm_logits, *kv_caches_out, *topk_outputs


def export_bloom_model(model, config, dtype, args, model_name):
    """
    Note
    # please be care of the format of kv cache
    # some models use format of [batch, head, seq_len, hidden_size]
    # while some models use format of [batch, seq_len, head, hidden_size]
    """
    onnx_file_name = os.path.join(args.out_dir, f"{model_name}.onnx")
    model_wrapper = BloomForCausalLMWrapper(model, config, args)

    hidden_size = config.hidden_size

    batch = 1
    N = 1
    sumN = 32
    lastN = sumN - N

    layer_num = config.n_layer

    input_ids = torch.ones([batch, N], dtype=torch.int64).to(args.device)
    attention_mask = torch.ones([1, sumN], dtype=torch.int64).to(args.device)

    in_names = ["input_ids", "attention_mask"]

    dynamic_axes = {
        'input_ids': {1: 'N', },
        'attention_mask': {1: "sumN"},
    }

    kv_caches_in = []
    out_names = ["lm_logits"]

    n_head = config.n_head
    cache_channel = hidden_size // n_head

    k_cache_in_shape = [n_head, cache_channel, lastN]
    v_cache_in_shape = [n_head, lastN, cache_channel]
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

    input_datas = (input_ids, attention_mask, kv_caches_in)

    if args.add_topk_warper > 0:
        out_names.extend(["logits_topk_value", "logits_topk_idx"])

    torch.onnx.export(
        model_wrapper,
        input_datas,
        onnx_file_name,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=in_names,
        output_names=out_names,
        dynamic_axes=dynamic_axes,
    )


def export_bloom(args):
    device = args.device
    if args.dtype == "float32":
        dtype = torch.float32
    elif args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16

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
    print("config:", config)

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

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    if args.dtype not in ["float32", "float16", "bfloat16"]:
        raise ValueError("dtype is invalid")

    export_bloom(args)
