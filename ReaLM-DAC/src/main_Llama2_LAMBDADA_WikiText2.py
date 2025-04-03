import torch
import numpy as np
import re
import json
import jsonlines
import os
import gc
import argparse

from torch import nn
from functools import partial
from smoothquant.error_inject import W8A8Linear, NoisyW8A8Linear, W8A8MatMul, NoisyW8A8MatMul
from datasets import load_dataset
from smoothquant.smooth import smooth_lm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation import GenerationConfig

from torch.utils.data import DataLoader
from transformers import TextDataset, DataCollatorForLanguageModeling
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import time

from sampling.autoregressive_sampling import autoregressive_sampling
import contexttimer
from rouge import Rouge
from rouge_score import rouge_scorer


def quantize_llama_model(model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=False):
    """
    Applies quantization to various layers in the Llama model.
    """
    from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP

    for name, m in model.model.named_modules():
        if isinstance(m, LlamaMLP):
            m.gate_proj = W8A8Linear.from_float(m.gate_proj, weight_quant=weight_quant, act_quant=act_quant)
            m.up_proj = W8A8Linear.from_float(m.up_proj, weight_quant=weight_quant, act_quant=act_quant)
            m.down_proj = W8A8Linear.from_float(m.down_proj, weight_quant=weight_quant, act_quant=act_quant)
        elif isinstance(m, LlamaAttention):
            m.q_proj = W8A8Linear.from_float(m.q_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.k_proj = W8A8Linear.from_float(m.k_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.v_proj = W8A8Linear.from_float(m.v_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.o_proj = W8A8Linear.from_float(m.o_proj, weight_quant=weight_quant, act_quant=act_quant)
            m.matmul1 = W8A8MatMul(act_quant=act_quant, quantize_output=False)
            m.matmul2 = W8A8MatMul(act_quant=act_quant, quantize_output=True)
    return model


def quantize_llama_model_error(model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=True, err_prob=0):
    """
    Applies error injection and quantization to the Llama model.
    """
    from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP
    i = 0
    for name, m in model.model.named_modules():
        if isinstance(m, LlamaMLP):
            if i == 0:
                m.gate_proj = NoisyW8A8Linear.from_float(m.gate_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob)
                m.up_proj = NoisyW8A8Linear.from_float(m.up_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob)
                m.down_proj = NoisyW8A8Linear.from_float(m.down_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob)
                i += 1
            else:
                m.gate_proj = W8A8Linear.from_float(m.gate_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
                m.up_proj = W8A8Linear.from_float(m.up_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
                m.down_proj = W8A8Linear.from_float(m.down_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
        elif isinstance(m, LlamaAttention):
            if i == 0:
                m.q_proj = NoisyW8A8Linear.from_float(m.q_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob)
                m.k_proj = NoisyW8A8Linear.from_float(m.k_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob)
                m.v_proj = NoisyW8A8Linear.from_float(m.v_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob)
                m.o_proj = NoisyW8A8Linear.from_float(m.o_proj, weight_quant=weight_quant, act_quant=act_quant, err_prob=err_prob)
                m.matmul1 = NoisyW8A8MatMul(act_quant=act_quant, quantize_output=False, err_prob=err_prob)
                m.matmul2 = NoisyW8A8MatMul(act_quant=act_quant, quantize_output=True, err_prob=err_prob)
            else:
                m.q_proj = W8A8Linear.from_float(m.q_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
                m.k_proj = W8A8Linear.from_float(m.k_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
                m.v_proj = W8A8Linear.from_float(m.v_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
                m.o_proj = W8A8Linear.from_float(m.o_proj, weight_quant=weight_quant, act_quant=act_quant)
                m.matmul1 = W8A8MatMul(act_quant=act_quant, quantize_output=False)
                m.matmul2 = W8A8MatMul(act_quant=act_quant, quantize_output=True)
    return model


class Evaluator:
    """
    Evaluates the model on the given dataset.
    """
    def __init__(self, dataset, tokenizer, device):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device
        self.dataset = self.dataset.map(lambda examples: self.tokenizer(examples['text']), batched=True)
        self.dataset.set_format(type='torch', columns=['input_ids'])

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        total, hit = 0, 0
        for batch in tqdm(self.dataset, desc="Evaluating"):
            input_ids = batch['input_ids'].to(self.device).unsqueeze(0)
            label = input_ids[:, -1]
            outputs = model(input_ids)
            last_token_logits = outputs.logits[:, -2, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
        accuracy = hit / total
        return round(accuracy * 100, 3)


class Evaluator_ppl:
    """
    Evaluates the perplexity (PPL) of the model on the given dataset.
    """
    def __init__(self, dataset, tokenizer, device, n_samples=40):
        self.dataset = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
        self.dataset = self.dataset.input_ids.to(device)
        self.n_samples = n_samples

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        nlls = []
        for i in tqdm(range(self.n_samples), desc="Evaluating..."):
            batch = self.dataset[:, (i * 2048):((i + 1) * 2048)].to(model.device)
            lm_logits = model(batch).logits
            shift_logits = lm_logits[:, :-1, :].contiguous().float()
            shift_labels = self.dataset[:, (i * 2048):((i + 1) * 2048)][:, 1:]
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            nlls.append(loss.float() * 2048)
        return torch.exp(torch.stack(nlls).sum() / (self.n_samples * 2048))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("-o", "--output_file", type=str, default="Llama_xsum_summary.jsonl")
    parser.add_argument("-r", "--input_file", type=str, default=None)
    args = parser.parse_args()

    err_prob_list = [0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    ppl_normal_list = []
    ppl_noisy_list = []
    acc_noisy_list = []
    x_sum_noisy_list = []

    start_time = time.time()
    output_summary = jsonlines.Writer(open(args.output_file, "w", encoding="utf-8"))

    for err_prob in err_prob_list:
        start_time_i = time.time()
        print(f"Error Probability: {err_prob}")

        print("Loading model...")
        model_fp32 = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf', torch_dtype=torch.float32, device_map='auto')
        act_scales = torch.load('act_scales/llama-2-7b.pt')

        print('Smoothing model...')
        smooth_lm(model_fp32, act_scales, 0.85)

        print('Loading tokenizer...')
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')

        print('Loading dataset...')
        dataset_lambada = load_dataset("lambada", split='validation[:1000]')
        dataset_wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

        evaluator = Evaluator(dataset_lambada, tokenizer, 'cuda')
        evaluator_ppl = Evaluator_ppl(dataset_wikitext, tokenizer, "cuda", n_samples=40)

        normal_model = quantize_llama_model(model_fp32)
        print('Evaluating...')
        acc = evaluator.evaluate(normal_model)
        ppl_normal = evaluator_ppl.evaluate(normal_model)

        print(f"Accuracy: {acc}, Perplexity: {ppl_normal}")

        end_time_i = time.time()
        print(f"Time for this iteration: {(end_time_i - start_time_i) / 60} minutes")

    end_time = time.time()
    print(f"Total time: {(end_time - start_time) / 60} minutes")
