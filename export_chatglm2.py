import os
import logging
import argparse
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
 
 
class ChatGLMModelWrapper(nn.Module):
    def __init__(self, chat_glm_model, config, args):
        super().__init__()
        self.chat_glm_model = chat_glm_model
        self.config = config
        self.max_seq_len = config.seq_length
        self.layer_num = chat_glm_model.encoder.num_layers
 
        self.args = args
 
    def forward(
        self, input_ids, attention_mask, position_ids, kv_caches
    ):
        # Rotary positional embeddings
        rotary_pos_emb = self.chat_glm_model.rotary_pos_emb(self.max_seq_len)
        rotary_pos_emb = rotary_pos_emb[position_ids]
        rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()
 
        # list to [(past_key, past_value) x layers]
        past_key_values = []
        for i in range(self.layer_num):
            past_key_values.append((kv_caches[2 * i], kv_caches[2 * i + 1]))
 
        inputs_embeds = self.chat_glm_model.embedding(input_ids)
 
        # Run encoder.
        hidden_states, presents, all_hidden_states, all_self_attentions = self.chat_glm_model.encoder(
            inputs_embeds, attention_mask, rotary_pos_emb=rotary_pos_emb,
            kv_caches=past_key_values, use_cache=True, output_hidden_states=False
        )
 
        kv_caches_out = []
        for layer_cache in presents:
            kv_caches_out.extend(list(layer_cache))
 
        hidden_states = hidden_states[-1:]
        lm_logits = self.chat_glm_model.output_layer(hidden_states)
 
        topk_outputs = []
        if self.args.add_topk_warper:
            logging.warning("add topk to glm model")
            if self.args.topk < 0:
                raise ValueError("topk {} is invalid")
            topk_outputs = torch.topk(lm_logits, k=self.args.topk, dim=-1)
 
        return lm_logits, *kv_caches_out, *topk_outputs
 
 
def export_chat_glm_model(chat_glm_model, config, dtype, args, model_name):
    """
    Note
    # please be care of the format of kv cache
    # some models use format of [batch, head, seq_len, hidden_size]
    # while some models use format of [batch, seq_len, head, hidden_size]
    """
    onnx_file_name = os.path.join(args.out_dir, f"{model_name}.onnx")
    glm_model_wrapper = ChatGLMModelWrapper(chat_glm_model, config, args)
 
    hidden_size = config.hidden_size
    layer_num = chat_glm_model.encoder.num_layers
    print("layer_num:", layer_num)
 
    kv_channels = config.kv_channels
 
    batch = 1
    N = 1
    sumN = 32
    lastN = sumN - N
 
    input_ids = torch.ones([batch, N], dtype=torch.int64).to(args.device)
    attention_mask = torch.zeros([batch, 1, N, sumN], dtype=torch.bool).to(args.device)
    position_ids = torch.ones([batch, N], dtype=torch.int64).to(args.device)
 
    in_names = ["input_ids", "attention_mask", "position_ids"]
 
    dynamic_axes = {
        'input_ids': {1: 'N'},
        'attention_mask': {2: 'N', 3: "sumN"},
        "position_ids": {1: 'N'},
    }
 
    kv_caches_in = []
    out_names = ["lm_logits"]
 
    kv_cache_in_shape = [lastN, 1, 2, kv_channels]
    kv_cache_dyn_axes = {0: "lastSum"}
 
    for i in range(layer_num):
        past_key_in = torch.randn(kv_cache_in_shape, dtype=dtype).to(args.device)
        past_value_in = torch.randn(kv_cache_in_shape, dtype=dtype).to(args.device)
 
        kv_caches_in.extend([past_key_in, past_value_in])
        in_names.extend([f"past_key_in{i}", f"past_value_in{i}"])
        out_names.extend([f"past_key{i}", f"past_value{i}"])
 
        dynamic_axes[f"past_key_in{i}"] = kv_cache_dyn_axes
        dynamic_axes[f"past_value_in{i}"] = kv_cache_dyn_axes
 
    input_datas = (input_ids, attention_mask, position_ids, kv_caches_in)
 
    if args.add_topk_warper:
        out_names.extend(["logits_topk_value", "logits_topk_idx"])
 
    torch.onnx.export(
        glm_model_wrapper,
        input_datas,
        onnx_file_name,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=in_names,
        output_names=out_names,
        dynamic_axes=dynamic_axes,
    )
 
 
def export_chatglm2(args):
    """
    this method convert embedding, transformer, output_layer into a single onnx model
    if you want to export them independently, please check older git commit
    """
    device = args.device
    if args.dtype == "float32":
        dtype = torch.float32
    elif args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
 
    print(f"begin load model from {args.model_path}")
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True)
    if args.dtype == "float32":
        model = model.float()
        print("convert model to float32")
    else:
        model = model.half()
        print("convert model to float16")
 
    if args.device == "cuda":
        model.cuda()
        print("convert model to cuda")
 
    model = model.eval()
 
    print(f"finish load model from {args.model_path}")
    config = model.config
 
    print("begin export chat_glm_model")
    chat_glm_model = model.transformer
    # chat_glm_model.encoder.num_layers = 2  # help debug
    export_chat_glm_model(chat_glm_model, config, dtype, args, "chat_glm_model")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='export chatglm2',
    )
    parser.add_argument('-m', '--model_path', required=True, type=str)
    parser.add_argument('-o', '--out_dir', required=False, type=str, default="")
    parser.add_argument('--opset', required=False, type=int, default=15)
    parser.add_argument('-d', '--device', required=False, type=str, choices=["cpu", "cuda"], default="cuda")
    parser.add_argument('-p', '--dtype', required=False, type=str, choices=["float32", "float16", "bfloat16"], default="float16")
 
    parser.add_argument('--add_topk_warper', action='store_true')
    parser.add_argument('--topk', required=False, type=int, default=4)
 
    args = parser.parse_args()
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    export_chatglm2(args)
