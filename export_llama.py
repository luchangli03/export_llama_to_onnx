import os
import argparse
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


def export_lm_head(lm_head_model, config, dtype, args, model_name):
    # fake size used to generate fake data
    batch = 1
    seq = 1
    hidden_size = config.hidden_size

    input_shape = [batch, seq, hidden_size]
    input_data = torch.randn(input_shape, dtype=dtype).to(args.device)

    onnx_file_name = os.path.join(args.out_dir, f"{model_name}.onnx")

    # Export the model
    torch.onnx.export(
        lm_head_model,
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


def export_norm(norm_model, config, dtype, args, model_name):
    batch = 1
    seq = 1
    hidden_size = config.hidden_size

    input_shape = [batch, seq, hidden_size]
    input_data = torch.randn(input_shape, dtype=dtype).to(args.device)

    onnx_file_name = os.path.join(args.out_dir, f"{model_name}.onnx")

    # Export the model
    torch.onnx.export(
        norm_model,
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


def export_embeding(embed_model, config, args, model_name):
    batch = 1
    seq = 1
    input_shape = [batch, seq]
    dtype = torch.int64
    input_data = torch.ones(input_shape, dtype=dtype).to(args.device)

    onnx_file_name = os.path.join(args.out_dir, f"{model_name}.onnx")

    # Export the model
    torch.onnx.export(
        embed_model,
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


class DecoderLayersWrapperLlama(nn.Module):
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


class DecoderLayersWrapperQwen(nn.Module):
    def __init__(self, layers, config):
        super().__init__()
        self.layers = layers
        self.config = config

    def forward(
        self,
        hidden_states,
        attention_mask,
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
                layer_past=[past_key_values_in[i * 2], past_key_values_in[i * 2 + 1]],
                output_attentions=output_attentions,
                use_cache=use_cache)

            past_key, past_value = kv_cache
            kv_caches_out.extend([past_key, past_value])
        return hidden_states, *kv_caches_out


def export_decoders(decoder_layers, config, dtype, args, model_name):
    """
    Note
    # please be care of the format of kv cache
    # some models use format of [batch, head, seq_len, hidden_size]
    # while some models use format of [batch, seq_len, head, hidden_size]
    """
    onnx_file_name = os.path.join(args.out_dir, f"{model_name}.onnx")

    hidden_size = config.hidden_size
    layer_num = len(decoder_layers)
    head_num = config.num_attention_heads
    hidden_size1 = hidden_size // head_num
    print("layer_num:", layer_num, hidden_size1)

    batch = 1
    N = 1
    sumN = 32
    lastN = sumN - N

    if not args.model_type:
        decoder_layers_wrapper = DecoderLayersWrapperLlama(decoder_layers, config)

        hidden_in = torch.randn([batch, N, hidden_size], dtype=dtype).to(args.device)
        attention_mask = torch.randn([batch, 1, N, sumN], dtype=dtype).to(args.device)
        position_ids = torch.ones([batch, N], dtype=torch.int64).to(args.device)

        in_names = ["hidden_in", "attention_mask", "position_ids"]

        dynamic_axes = {
            'hidden_in': {1: 'N', },
            'attention_mask': {1: 'N', 2: "sumN"},
            "position_ids": {1: 'N', },
        }

    elif args.model_type == "Qwen":
        decoder_layers_wrapper = DecoderLayersWrapperQwen(decoder_layers, config)

        hidden_in = torch.randn([batch, N, hidden_size], dtype=dtype).to(args.device)
        attention_mask = torch.randn([batch, 1, 1, sumN], dtype=dtype).to(args.device)

        in_names = ["hidden_in", "attention_mask"]

        dynamic_axes = {
            'hidden_in': {1: 'N', },
            'attention_mask': {1: 'N', 2: "sumN"},
        }

    kv_caches_in = []
    out_names = ["hidden_out"]

    kv_cache_in_shape = [batch, head_num, lastN, hidden_size1]
    kv_cache_dyn_axes = {2: "lastSum"}
    if args.kv_cache_format == 1:
        kv_cache_in_shape = [batch, lastN, head_num, hidden_size1]
        kv_cache_dyn_axes = {1: "lastSum"}

    for i in range(layer_num):
        past_key_in = torch.randn(kv_cache_in_shape, dtype=dtype).to(args.device)
        past_value_in = torch.randn(kv_cache_in_shape, dtype=dtype).to(args.device)

        kv_caches_in.extend([past_key_in, past_value_in])
        in_names.extend([f"past_key_in{i}", f"past_value_in{i}"])
        out_names.extend([f"past_key{i}", f"past_value{i}"])

        dynamic_axes[f"past_key_in{i}"] = kv_cache_dyn_axes
        dynamic_axes[f"past_value_in{i}"] = kv_cache_dyn_axes

    if not args.model_type:
        input_datas = (hidden_in, attention_mask, position_ids, kv_caches_in)
    elif args.model_type == "Qwen":
        input_datas = (hidden_in, attention_mask, kv_caches_in)

    torch.onnx.export(
        decoder_layers_wrapper,
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
    dtypes_config = {
        "fp32": False,
        "fp16": False,
        "bf16": False,
    }
    if args.dtype == "float32":
        dtype = torch.float32
        dtypes_config["fp32"] = True
    elif args.dtype == "float16":
        dtype = torch.float16
        dtypes_config["fp16"] = True
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
        dtypes_config["bf16"] = True

    print(f"begin load model from {args.model_path}")
    # tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    if not args.model_type:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, device_map=device, torch_dtype=dtype, trust_remote_code=True).eval()
    elif args.model_type == "Qwen":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, device_map=device, **dtypes_config, trust_remote_code=True).eval()

    print(f"finish load model from {args.model_path}")
    config = model.config

    if not args.model_type:
        # default configure for llama like models
        lm_head_model = model.lm_head
        embeding_model = model.model.embed_tokens
        norm_model = model.model.norm
        decoder_layers = model.model.layers
    elif args.model_type == "Qwen":
        # support alibaba Qwen
        lm_head_model = model.lm_head
        embeding_model = model.transformer.wte
        norm_model = model.transformer.ln_f
        decoder_layers = model.transformer.h
        args.kv_cache_format = 1
    else:
        raise ValueError("invalid model_type")

    print(f"begin export_lm_head")
    export_lm_head(lm_head_model, config, dtype, args, "lm_head")

    print(f"begin export_embeding")
    export_embeding(embeding_model, config, args, "embeding")

    print(f"begin export_norm")
    export_norm(norm_model, config, dtype, args, "norm")

    print(f"begin export_decoders")
    decoder_pack_size = args.decoder_pack_size
    if decoder_pack_size <= 0:
        # export decoders as one onnx models
        export_decoders(decoder_layers, config, dtype, args, "decoders")
    else:
        # export decoders to multiple onnx models
        decoder_num = len(decoder_layers)
        export_model_num = (decoder_num + decoder_pack_size - 1) // decoder_pack_size

        for i in range(export_model_num):
            layers = decoder_layers[i * decoder_pack_size:(i + 1) * decoder_pack_size]
            export_decoders(layers, config, dtype, args, f"decoders_{i}")


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
    # 0: export all decoders into one onnx. >0: export multiple onnx files, and each onnx has decoder_pack_size layers
    parser.add_argument('--decoder_pack_size', required=False, type=int, default=0)
    # 0 means [batch, head, seq, hidden], 1 means [batch, seq, head, hidden]
    parser.add_argument('--kv_cache_format', required=False, type=int, default=0)
    # default model_type is llama_hf, other model_type such as Qwen, Baichuan ban be supported
    parser.add_argument('--model_type', required=False, type=str, default="")

    args = parser.parse_args()

    if args.dtype not in ["float32", "float16", "bfloat16"]:
        raise ValueError("dtype is invalid")

    export_llama(args)
