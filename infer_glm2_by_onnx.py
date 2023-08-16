import numpy as np
from onnx_rt_utils import OnnxRuntimeModel, get_random_data
from sample_utils import sample_topk
from transformers import AutoTokenizer


def gen_attention_mask(N, sumN):
    attention_mask = np.zeros(shape=(N, sumN), dtype="bool")
    for i in range(N):
        mask_num = N - 1 - i
        start = sumN - mask_num
        for j in range(start, sumN):
            attention_mask[i, j] = True
    return attention_mask


def prepare_kv_cache_round0(glm_model_inputs, layer_num, lastSum):
    """
    only used at the first time
    """
    for i in range(layer_num):
        past_key_in = get_random_data([lastSum, 1, 2, 128], "float16")
        past_value_in = get_random_data([lastSum, 1, 2, 128], "float16")

        past_key_in_name = f"past_key_in{i}"
        past_value_in_name = f"past_value_in{i}"

        glm_model_inputs[past_key_in_name] = past_key_in
        glm_model_inputs[past_value_in_name] = past_value_in
    return glm_model_inputs


def prepare_kv_cache_from_outputs(glm_model_inputs, decoder_outputs, layer_num):
    offset = 1
    for i in range(layer_num):
        past_key_in_name = f"past_key_in{i}"
        past_value_in_name = f"past_value_in{i}"

        glm_model_inputs[past_key_in_name] = decoder_outputs[offset + i * 2]
        glm_model_inputs[past_value_in_name] = decoder_outputs[offset + i * 2 + 1]
    return glm_model_inputs


# all decoder layer num
layer_num = 28
eos_token_id = 2

query = "你好"

round_id = 1
prompt = "[Round {}]\n\n问：{}\n\n答：".format(round_id, query)

# you can simply replace AutoTokenizer by sentencepiece tokenizer and manually add spetial tokens
model_path = "/mnt/f/models/chatglm2-6b/"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

input_ids = tokenizer(prompt)['input_ids']
print(input_ids)

input_ids = np.array(input_ids).reshape([1, -1]).astype("int64")
N = input_ids.shape[1]
sumN = N
lastSum = sumN - N
print("N:", N, sumN, lastSum)

position_ids = np.arange(sumN).reshape([1, -1]).astype("int64")

input_ids = input_ids.astype("int64")
position_ids = position_ids.astype("int64")

glm_model = OnnxRuntimeModel("build/chat_glm_model.onnx")

max_seq = 512

glm_model_inputs = {}

gen_tokens = []

for i in range(max_seq):
    print("input_ids:", input_ids)
    print("position_ids:", position_ids)

    attention_mask = gen_attention_mask(N, sumN).astype("bool")
    print("attention_mask:", attention_mask)
    attention_mask = attention_mask.reshape([1, 1, N, sumN])

    glm_model_inputs["input_ids"] = input_ids
    glm_model_inputs["attention_mask"] = attention_mask
    glm_model_inputs["position_ids"] = position_ids

    if i == 0:
        glm_model_inputs = prepare_kv_cache_round0(glm_model_inputs, layer_num, lastSum)

    glm_model_outputs = glm_model(**glm_model_inputs)
    lm_logits = glm_model_outputs[0]
    print("lm_logits:", lm_logits)

    next_token = sample_topk(lm_logits, topk=1)
    gen_tokens.append(next_token)
    print("next_token:", next_token)

    if next_token == eos_token_id:
        break

    input_ids = np.array([next_token]).astype("int64").reshape([-1, 1])
    position_ids = np.array([sumN]).astype("int64").reshape([-1, 1])
    N = 1
    sumN += 1

    prepare_kv_cache_from_outputs(glm_model_inputs, glm_model_outputs, layer_num)

gen_text = tokenizer.decode(gen_tokens)
print("Q:", query)
print("A:", gen_text)
