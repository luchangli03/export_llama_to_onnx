from transformers import AutoTokenizer
model_path = "/mnt/f/models/chatglm2-6b/"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


gen_tokens=[64790, 64792 ,  790 ,30951 ,  517, 30910 ,30939, 30996 ,   13   , 13 ,54761, 31211,
  39701 ,   13  ,  13 ,55437 ,31211]

gen_text = tokenizer.decode(gen_tokens)
print("gen_text:", gen_text)

