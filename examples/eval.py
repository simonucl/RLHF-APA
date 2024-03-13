import argparse
import json
import os
import torch
import tqdm
import numpy as np
from trlx.models.modeling_sppo import (
    AdaptiveKLController,
    AutoModelForCausalLMWithHydraValueHead,
    AutoModelForSeq2SeqLMWithHydraValueHead,
    FixedKLController,
)
from transformers import  AutoTokenizer

from countdown_utils import *

parser = argparse.ArgumentParser()

parser.add_argument('--ckpt', type=str, default='ckpt/ckpt_best.pt')
parser.add_argument('--pt', type=str, default='ckpt/ckpt_best.pt')
parser.add_argument("-n", "--num",type=int, default=10)
parser.add_argument("-o", "--offset",type=int, default=0)
parser.add_argument("--data_dir", type=str, default="/scr/kanishkg/rational-cot/data/")
parser.add_argument("-d", "--data",type=str, default="val_b3_t100_n100000_random.json")
parser.add_argument("-b", "--batch_size",type=int, default=1)
parser.add_argument("-c", "--ctx",type=int, default=4096)
parser.add_argument("-t", "--temperature",type=float, default=0.0)

def eval_ll(model, tokenizer, data, batch_size=128, context_len=4096, temperature=0.0, n=1):
    """
    Evaluate the model on the data using a sliding window so that the context length is not exceeded
    """
    output_texts_concat = []
    for b in tqdm.trange(0, len(data), batch_size):
        batch = data[b:min(b+batch_size, len(data))]
        output_texts = ["" for _ in range(len(batch))]
        tokenizer.padding_side = "left"
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to("cuda")
        inputs = inputs['input_ids']
        # with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):

        if n == 1:
            if temperature == 0.0:
                outputs = model.generate(input_ids=inputs, pad_token_id=tokenizer.eos_token_id, attention_mask=torch.ones_like(inputs), max_length=context_len, num_beams=1, do_sample=False)
            else:
                outputs = model.generate(input_ids=inputs, pad_token_id=tokenizer.eos_token_id, attention_mask=torch.ones_like(inputs), max_length=context_len, num_beams=1, do_sample=True, temperature=temperature)
            # split output vector into first N tokens and the rest
            output_tokens = outputs
            output_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=False)
            tokenizer.padding_side = "left"
            output_texts = [ot + ot_now for ot, ot_now in zip(output_texts, output_text)]
            # print token lens of tokenized outputs
            print([len(tokenizer(ot)['input_ids']) for ot in output_texts])
            output_texts_concat += output_texts
    return output_texts_concat


args = parser.parse_args()

model = AutoModelForCausalLMWithHydraValueHead.from_pretrained('/scr/kanishkg/rational-cot/models/sft-mix-4-cd5e5/checkpoint-45500')
state_dict = torch.load('/scr/kanishkg/trl/outputs/checkpoint_01000/pytorch_model/mp_rank_00_model_states.pt')

model.post_init(state_dict=state_dict)

model.to("cuda")
model.eval()
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B', padding_side='left')
tokenizer.pad_token = tokenizer.eos_token

data_file = os.path.join(args.data_dir, args.data)
with open(data_file, "r") as json_file:
    data = json.load(json_file)

predictions = []
pred_ratings = []
pred_reasons = []
tokenizer.padding_side = "left"
test_prompts = [tokenizer.bos_token + f"Current State: {sample['target']}:{sample['nums']}, Operations: []"  for sample in data[args.offset:args.num]]
len_nums = [len(sample['nums']) for sample in data[args.offset:args.num]]
data_4 = [d for d, l in zip(test_prompts, len_nums) if l == 4]


comp_4 = eval_ll(model, tokenizer, data_4, batch_size=args.batch_size, context_len=args.ctx, temperature=args.temperature, n=1)
predictions = comp_4

# rate outputs
true_rating = []
for i in range(len(predictions)):
    rating, reason = metric_fn(predictions[i].split(tokenizer.bos_token)[1], mode="sft")
    tr, _ = metric_fn(f"{data[i]['search_path']}", mode="sft")
    pred_ratings.append(rating)
    true_rating.append(tr)
    pred_reasons.append(reason)

# print results
print("Results Summary:")
print(f"Average rating: {np.mean(pred_ratings)}")
print(f"Average true rating: {np.mean(true_rating)}")
# print accuracy ie rating > 0
print(f"Accuracy: {np.mean([r > 0 for r in pred_ratings])}")
print(f"True Accuracy: {np.mean([r > 0 for r in true_rating])}")

