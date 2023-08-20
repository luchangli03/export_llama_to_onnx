import os
import argparse
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class LlamaForCausalLMWrapper(nn.Module):
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
        inputs_embeds = self.model.model.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        kv_caches_out = []
        for idx, decoder_layer in enumerate(self.model.model.layers):
            past_key_value = past_key_values[idx]
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]
            kv_caches_out.extend(layer_outputs[2 if output_attentions else 1],)

        hidden_states = hidden_states[:, -1, :]
        hidden_states = self.model.model.norm(hidden_states)
        lm_logits = self.model.lm_head(hidden_states)

        topk_outputs = []
        if self.args.add_topk_warper > 0:
            logging.warning("add topk to glm model")
            if self.args.topk < 0:
                raise ValueError("topk {} is invalid")
            topk_outputs = torch.topk(lm_logits, k=self.args.topk, dim=-1)

        return lm_logits, *kv_caches_out, *topk_outputs


def export_llama_to_single_onnx(model, config, dtype, args, model_name):
    llama_model_wrapper = LlamaForCausalLMWrapper(model, config, args)

    onnx_file_name = os.path.join(args.out_dir, f"{model_name}.onnx")

    layer_num = len(model.model.layers)

    hidden_size = config.hidden_size
    head_num = config.num_attention_heads
    hidden_size1 = hidden_size // head_num

    batch = 1
    N = 1
    sumN = 32
    lastN = sumN - N

    input_ids_shape = [batch, N]
    input_ids = torch.ones(input_ids_shape, dtype=torch.int64).to(args.device)
    attention_mask = torch.randn([batch, 1, N, sumN], dtype=dtype).to(args.device)
    position_ids = torch.ones([batch, N], dtype=torch.int64).to(args.device)

    in_names = ["hidden_in", "attention_mask", "position_ids"]

    dynamic_axes = {
        'hidden_in': {1: 'N', },
        'attention_mask': {2: 'N', 3: "sumN"},
        "position_ids": {1: 'N', },
    }

    kv_caches_in = []
    out_names = ["hidden_out"]

    kv_cache_in_shape = [batch, head_num, lastN, hidden_size1]
    kv_cache_dyn_axes = {2: "lastSum"}

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
    if args.dtype == "float32":
        dtype = torch.float32
    elif args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16

    print(f"begin load model from {args.model_path}")
    # tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, device_map=device, torch_dtype=dtype, trust_remote_code=True).eval()

    print(f"finish load model from {args.model_path}")
    config = model.config
    print("config:", config)

    print(f"begin export llama")
    export_llama_to_single_onnx(model, config, dtype, args, "llama_onnx")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='export llama',
    )
    parser.add_argument('-m', '--model_path', required=True, type=str)
    parser.add_argument('-o', '--out_dir', required=False, type=str, default="")
    parser.add_argument('--opset', required=False, type=int, default=15)
    parser.add_argument('-d', '--device', required=False, type=str, default="cuda")
    # supported dtype: ["float32", "float16", "bfloat16"]
    parser.add_argument('-p', '--dtype', required=False, type=str, default="float16")
    parser.add_argument('--add_topk_warper', required=False, type=int, default=0)
    parser.add_argument('--topk', required=False, type=int, default=4)

    args = parser.parse_args()

    if args.dtype not in ["float32", "float16", "bfloat16"]:
        raise ValueError("dtype is invalid")

    export_llama(args)
