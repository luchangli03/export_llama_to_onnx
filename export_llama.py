import os
import argparse
import torch
from torch import nn
from transformers import AutoTokenizer, LlamaForCausalLM


def export_lm_head(model, config, dtype, args):
    batch = 1
    seq = 1
    hidden_size = config.hidden_size

    input_shape = [batch, seq, hidden_size]
    input_data = torch.randn(input_shape, dtype=dtype).to(args.device)

    onnx_file_name = os.path.join(args.out_dir, "lm_head1.onnx")

    # Export the model
    torch.onnx.export(
        model.lm_head,
        input_data,
        onnx_file_name,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {1: 'N'}
        },
    )


def export_norm(model, config, dtype, args):
    batch = 1
    seq = 1
    hidden_size = config.hidden_size

    input_shape = [batch, seq, hidden_size]
    input_data = torch.randn(input_shape, dtype=dtype).to(args.device)

    onnx_file_name = os.path.join(args.out_dir, "norm.onnx")

    # Export the model
    torch.onnx.export(
        model.model.norm,
        input_data,
        onnx_file_name,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {1: 'N'}
        },
    )


def export_embeding(model, config, args):
    batch = 1
    seq = 1
    input_shape = [batch, seq]
    dtype = torch.int64
    input_data = torch.ones(input_shape, dtype=dtype).to(args.device)

    onnx_file_name = os.path.join(args.out_dir, "embeding.onnx")

    # Export the model
    torch.onnx.export(
        model.model.embed_tokens,
        input_data,
        onnx_file_name,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {1: 'N'}
        },
    )


class DecoderLayersWrapper(nn.Module):
    def __init__(self, layers, config):
        super().__init__()
        self.layers = layers
        self.config = config

    def forward(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        # past_key_values_in format is [key0, value0, key1, value1, etc.]
        past_key_values_in,
        output_attentions=False,
        use_cache=True,
    ):
        kv_caches_out = []
        layer_num = len(self.layers)
        for i in range(layer_num):
            layer = self.layers[i]
            hidden_states, kv_cache = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=[past_key_values_in[i * 2], past_key_values_in[i * 2 + 1]],
                output_attentions=output_attentions,
                use_cache=use_cache)

            past_key, past_value = kv_cache
            kv_caches_out.extend([past_key, past_value])
        return hidden_states, *kv_caches_out


def export_decoders(model, config, dtype, args):
    """
    Note
    # please be care of the format of kv cache
    # some models use format of [batch, head, seq_len, hidden_size]
    # while some models use format of [batch, seq_len, head, hidden_size]
    """
    onnx_file_name = os.path.join(args.out_dir, "decoders.onnx")

    hidden_size = config.hidden_size

    batch = 1
    N = 1
    sumN = 32
    lastN = sumN - N

    layers = model.model.layers
    layer_num = len(layers)

    head_num = config.num_attention_heads
    hidden_size1 = hidden_size // head_num

    print("layer_num:", layer_num, hidden_size1)

    layers0_wrapper = DecoderLayersWrapper(layers, config)

    hidden_in = torch.randn([batch, N, hidden_size], dtype=dtype).to(args.device)
    attention_mask = torch.randn([batch, 1, N, sumN], dtype=dtype).to(args.device)
    position_ids = torch.ones([batch, N], dtype=torch.int64).to(args.device)

    kv_caches_in = []
    in_names = ["hidden_in", "attention_mask", "position_ids"]
    out_names = ["hidden_out"]

    dynamic_axes = {
        'hidden_in': {1: 'N', },
        'attention_mask': {1: 'N', 2: "sumN"},
        "position_ids": {1: 'N', },
    }

    for i in range(layer_num):
        past_key_in = torch.randn([batch, head_num, lastN, hidden_size1], dtype=dtype).to(args.device)
        past_value_in = torch.randn([batch, head_num, lastN, hidden_size1], dtype=dtype).to(args.device)

        kv_caches_in.extend([past_key_in, past_value_in])
        in_names.extend([f"past_key_in{i}", f"past_value_in{i}"])
        out_names.extend([f"past_key{i}", f"past_value{i}"])

        dynamic_axes[f"past_key_in{i}"] = {2: "lastN"}
        dynamic_axes[f"past_value_in{i}"] = {2: "lastN"}

    torch.onnx.export(
        layers0_wrapper,
        (hidden_in, attention_mask, position_ids, kv_caches_in),
        onnx_file_name,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=in_names,
        output_names=out_names,
        dynamic_axes=dynamic_axes,
    )


def export_llama(args):
    device = args.device
    dtype = torch.float32
    if args.dtype == "float16":
        dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = LlamaForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype, device_map=device,
    )
    config = model.config

    # generation test
    # prompt = 'Q: What is the largest animal?\nA:'
    # input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

    # generation_output = model.generate(
    #     input_ids=input_ids, max_new_tokens=128
    # )
    # print(tokenizer.decode(generation_output[0]))

    # export_lm_head(model, config, dtype, args)
    # export_norm(model, config, dtype, args)
    # export_embeding(model, config, args)
    export_decoders(model, config, dtype, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='export llama',
    )
    parser.add_argument('-m', '--model_path', required=True, type=str)
    parser.add_argument('-o', '--out_dir', required=False, type=str, default="")
    parser.add_argument('--opset', required=False, type=int, default=15)
    parser.add_argument('-d', '--device', required=False, type=str, default="cuda")
    parser.add_argument('-p', '--dtype', required=False, type=str, default="float16")

    args = parser.parse_args()

    export_llama(args)
