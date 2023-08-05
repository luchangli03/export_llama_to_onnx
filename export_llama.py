import os
import argparse
import torch
from transformers import AutoTokenizer, LlamaForCausalLM


def export_lm_head(model, config, args):
    batch = 1
    seq = 1
    hidden_size = config.hidden_size

    dtype = torch.float16
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

    export_lm_head(model, config, args)


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
