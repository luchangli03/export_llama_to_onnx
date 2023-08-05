import torch
from transformers import AutoTokenizer, LlamaForCausalLM
import argparse


def export_llama(model_path, out_dir="", opset=15, device="cuda"):

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map=device,
    )

    # generation test
    # prompt = 'Q: What is the largest animal?\nA:'
    # input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

    # generation_output = model.generate(
    #     input_ids=input_ids, max_new_tokens=128
    # )
    # print(tokenizer.decode(generation_output[0]))

    config = model.config

    batch = 1
    seq = 1
    hidden_size = config.hidden_size

    dtype = torch.float16
    input_shape = [batch, seq, hidden_size]
    input_data = torch.randn(input_shape, dtype=dtype).to(device)

    # Export the model
    torch.onnx.export(
        model.lm_head,
        input_data,
        "lm_head.onnx",
        opset_version=15,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {1: 'N'}
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='export llama',
    )
    parser.add_argument('-m', '--model_path', required=True, type=str)
    parser.add_argument('-o', '--out_dir', required=False, type=str)
    parser.add_argument('--opset', required=False, type=int, default=15)
    parser.add_argument('-d', '--device', required=False, type=str, default="cuda")

    args = parser.parse_args()

    export_llama(model_path=args.model_path, out_dir=args.out_dir, opset=args.opset, device=args.device)
