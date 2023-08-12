

import numpy as np
 
 
def sample_topk(tensor, topk=3):
    """
    only support batch=1 input
    """
    tensor = tensor.reshape([-1]).astype("float32")
    topk_vals, topk_idxs = warp_topk1(tensor, topk)
    probs = npsoftmax(topk_vals, axis=0)
    max_idx = np.random.multinomial(1, probs).argmax()
    next_token = topk_idxs[max_idx]
    return next_token
 
 
def sample_no_warp(lm_logits):
    """
    should use top-p or top-k warp processor
    """
    lm_logits = lm_logits.reshape([-1, lm_logits.shape[-1]])
    lm_logits = lm_logits.astype("float64")  # not necessary for cpp
    probs = npsoftmax(lm_logits, -1)
    next_token = npmultinominal2D(probs)
    return next_token
 
 
def npsoftmax(x, axis):
    y = x - np.max(x, axis=axis, keepdims=True)
    return np.exp(y) / np.sum(np.exp(y), axis=axis, keepdims=True)
 
 
def npmultinominal2D(x):
    ret = []
    for row, pval in enumerate(x):
        next_tok = np.random.multinomial(1, pval).argmax()
        ret.append(next_tok)
    return np.array(ret).astype("int64")
 
 
def warp_topk1(tensor, topk):
    tensor_1d = tensor.reshape([-1])
    topk_vals, topk_idxs = get_topk(tensor_1d, topk=topk)
    return topk_vals, topk_idxs
 
 
def get_topk(tensor_1d, topk=3):
    # value in topk_vals are placed by descending order
    topk_vals = [-float("Inf")] * topk
    topk_idxs = [0] * topk
 
    for idx, elem in enumerate(tensor_1d):
        if elem > topk_vals[topk - 1]:
            for i in range(topk):
                # find where current top value should be placed
                # then we right shift the topk_vals to place the top value
                if elem > topk_vals[i]:
                    # right shift
                    for j in reversed(range(i, topk-1)):
                        topk_vals[j+1] = topk_vals[j]
                        topk_idxs[j+1] = topk_idxs[j]
 
                    topk_vals[i] = elem
                    topk_idxs[i] = idx
                    break
    return topk_vals, topk_idxs