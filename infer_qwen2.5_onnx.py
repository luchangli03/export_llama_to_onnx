import numpy as np
from onnx_rt_utils import OnnxRuntimeModel
from sample_utils import sample_topk
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_2d_causal_mask(N, sumN, padded_len):
    attention_mask = np.zeros(N * padded_len)
    min_val = -65504.0

    pad_num = padded_len - sumN
    if (N == sumN):
        if (sumN != padded_len):
            raise ValueError("sumN != padded_len when N != 1\n")

        for i in range(N):
            mask_num = N - 1 - i + pad_num
            start = padded_len - mask_num
            for j in range(start, padded_len):
                attention_mask[i * padded_len + j] = min_val
    else:
        if (N != 1):
            raise ValueError("N is not 1\n")

        for i in range(sumN, padded_len):
            attention_mask[i] = min_val
    attention_mask = attention_mask.astype("float16")
    return attention_mask


def prepare_kv_cache_round0(model_inputs, layer_num, lastSum):
    """
    only used at the first time
    """
    for i in range(layer_num):
        past_key_in = np.zeros(shape=[1, 2, lastSum, 64], dtype="float16")
        past_value_in = np.zeros(shape=[1, 2, lastSum, 64], dtype="float16")

        past_key_in_name = f"past_key_in{i}"
        past_value_in_name = f"past_value_in{i}"

        model_inputs[past_key_in_name] = past_key_in
        model_inputs[past_value_in_name] = past_value_in
    return model_inputs


def prepare_kv_cache_from_outputs(model_inputs, decoder_outputs, layer_num):
    offset = 1
    for i in range(layer_num):
        past_key_in_name = f"past_key_in{i}"
        past_value_in_name = f"past_value_in{i}"

        model_inputs[past_key_in_name] = decoder_outputs[offset + i]
        model_inputs[past_value_in_name] = decoder_outputs[offset + layer_num + i]
    return model_inputs


layer_num = 24
eos_token_id = 151645

model_name = 'qwen2.5-0.5B'
tokenizer = AutoTokenizer.from_pretrained(model_name)

input_ids = [
]
print(input_ids)

input_ids = np.array(input_ids).reshape([1, -1]).astype("int64")
N = input_ids.shape[1]
sumN = N
lastSum = sumN - N
print("N:", N, sumN, lastSum)

position_ids = np.arange(sumN).reshape([1, -1]).astype("int64")

input_ids = input_ids.astype("int64")
position_ids = position_ids.astype("int64")

glm_model = OnnxRuntimeModel("qwen2.5_0.5b_onnx.infer.onnx")

max_seq = 512

gen_tokens = []
model_inputs = {}

for i in range(max_seq):
    print("input_ids:", input_ids)
    print("position_ids:", position_ids)

    attention_mask = get_2d_causal_mask(N, sumN, sumN)
    print("attention_mask:", attention_mask)
    attention_mask = attention_mask.reshape([1, 1, N, sumN])

    model_inputs["input_ids"] = input_ids
    model_inputs["attention_mask"] = attention_mask
    model_inputs["position_ids"] = position_ids

    if i == 0:
        model_inputs = prepare_kv_cache_round0(model_inputs, layer_num, lastSum)

    model_outputs = glm_model(**model_inputs)
    lm_logits = model_outputs[0]
    print("lm_logits:", lm_logits)

    next_token = sample_topk(lm_logits, topk=1)

    next_token = model_outputs[-1].reshape([-1])[0]

    gen_tokens.append(next_token)
    print("next_token:", next_token)

    if next_token == eos_token_id:
        break

    input_ids = np.array([next_token]).astype("int64").reshape([-1, 1])
    position_ids = np.array([sumN]).astype("int64").reshape([-1, 1])
    N = 1
    sumN += 1

    prepare_kv_cache_from_outputs(model_inputs, model_outputs, layer_num)

print("gen_tokens:", gen_tokens)
response = tokenizer.batch_decode([gen_tokens], skip_special_tokens=True)[0]
print("response:", response)
