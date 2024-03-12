import json
import math
import os
import sys
import argparse

import numpy as np
import torch
import tritonclient.grpc as client_util
from datasets import load_dataset
from huggingface_hub import snapshot_download
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTNeoForCausalLM
from tritonclient.utils import np_to_triton_dtype

from countdown_utils import *

import trlx
from trlx.data.default_configs import (
    ModelConfig,
    OptimizerConfig,
    SPPOConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)
import random

# parser = argparse.ArgumentParser(description='Train with APA.')
# parser.add_argument('--ckpt', type=str, default=None, help='Path to the checkpoint to load')

RANDOM_SEED = 0
LOSS = "square" # "square" or "log", square for APA and log for AWR
ADV_COEFF_SQ = 10
ADV_COEFF_LOG = 0.5
OUTPUT_DIR = "scr/kanishkg/trl/outputs"

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED) 



# config_name = MODEL_SIZE 
# if config_name == "125M":
#     default_config.train.batch_size = 8
#     default_config.method.chunk_size = 16
#     default_config.train.total_steps = 20000
#     default_config.model.model_path = "Dahoas/pythia-125M-static-sft"
#     default_config.tokenizer.tokenizer_path = "EleutherAI/gpt-neox-20b"
#     default_config.method.num_rollouts = 128
# elif config_name == "1B":
#     default_config.train.batch_size = 2
#     default_config.train.total_steps = 20000
#     default_config.model.model_path = "Dahoas/pythia-1B-static-sft"
#     default_config.tokenizer.tokenizer_path = "EleutherAI/gpt-neox-20b"
#     default_config.method.chunk_size = 4
# elif config_name == "6B":
#     default_config.train.batch_size = 1
#     default_config.train.total_steps = 20000
#     default_config.model.model_path = "Dahoas/pythia-6B-static-sft" # "databricks/dolly-v2-7b" #  
#     default_config.tokenizer.tokenizer_path =  "EleutherAI/gpt-neox-20b" # "databricks/dolly-v2-7b" #
#     default_config.method.chunk_size = 1 
# elif config_name == "20B":
#     default_config.train.seq_length = 512
#     default_config.train.batch_size = 1
#     default_config.train.total_steps = 8000
#     default_config.model.model_path = "EleutherAI/gpt-neox-20b"
#     default_config.tokenizer.tokenizer_path = "EleutherAI/gpt-neox-20b"
#     default_config.method.num_rollouts = 16
#     default_config.method.chunk_size = 4
#     default_config.method.ppo_epochs = 2


def prepare_tensor(name: str, input):
    t = client_util.InferInput(name, input.shape, np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t


def main(hparams={}):
    output_dir = OUTPUT_DIR
    default_config = TRLConfig(
    train=TrainConfig(
        seq_length=4096,
        epochs=10000,
        total_steps=20000,
        batch_size=2,
        checkpoint_interval=500,
        eval_interval=500,
        pipeline="PromptPipeline",
        trainer="AccelerateSPPOTrainer",
        checkpoint_dir="/scr/kanishkg/trl/checkpoints/apa_plan",
        seed=RANDOM_SEED,
    ),
    model=ModelConfig(model_path='/scr/kanishkg/rational-cot/models/sft-mix-4-cd5e5/checkpoint-45500/', num_layers_unfrozen=-1),
    tokenizer=TokenizerConfig(tokenizer_path="EleutherAI/gpt-neo-1.3B", padding_side="left"),
    optimizer=OptimizerConfig(name="adamw", kwargs=dict(lr=1e-6, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)),
    scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=10000, eta_min=1e-6)),
    method=SPPOConfig(
        name="SPPOConfig",
        num_rollouts=1,
        chunk_size=1,
        ppo_epochs=2,
        init_kl_coef=0.05,
        target=6,
        horizon=10000,
        gamma=1,
        lam=0.95,
        cliprange=100,
        cliprange_value=100,
        vf_coef=1,
        scale_reward="running",
        ref_mean=None,
        ref_std=None,
        loss_str=LOSS,
        adv_coeff_sq=ADV_COEFF_SQ,
        adv_coeff_log=ADV_COEFF_LOG,        
        cliprange_reward=100,
        gen_kwargs=dict(
            max_new_tokens=4000,
            do_sample=True,
            temperature=0.8,
        ),
    ),
)
    config = TRLConfig.update(default_config, hparams)
    config.train.checkpoint_dir = output_dir
    config.train.logging_dir = output_dir
    config.train.tracker = "wandb"

    # model = GPTNeoForCausalLM.from_pretrained(args.ckpt, torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2')
    # model.eval()
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    data_file = os.path.join('/scr/kanishkg/rational-cot/data/b4_3_random', 'train1_b4_t100_n500000_random.json')
    with open(data_file, "r") as json_file:
        data = json.load(json_file)
    prompts = [tokenizer.bos_token + f"Current State: {sample['target']}:{sample['nums']}, Operations: []"  for sample in data]

    val_file = os.path.join('/scr/kanishkg/rational-cot/data/b4_3_random', 'val1_b4_t100_n500000_random.json')
    with open(val_file, "r") as json_file:
        val_data = json.load(json_file)
    prompts = [tokenizer.bos_token + f"Current State: {sample['target']}:{sample['nums']}, Operations: []"  for sample in val_data]
    eval_prompts = prompts[:100]

    trlx.train(
        prompts=prompts,
        eval_prompts=eval_prompts,
        reward_fn=reward_fn,
        config=config,
        stop_sequences=["assistant:"],
    )


if __name__ == "__main__":

    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)
