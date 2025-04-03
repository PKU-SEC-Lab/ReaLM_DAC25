import os
import re
import pdb
import time
import json
import torch
import numpy as np
import contexttimer

from functools import partial
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from rouge import Rouge
from rouge_score import rouge_scorer
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    TextDataset,
    DataCollatorForLanguageModeling,
)

from sampling.autoregressive_sampling import autoregressive_sampling
from smoothquant.smooth import smooth_lm
from smoothquant.error_inject import (
    W8A8Linear,
    NoisyW8A8Linear,
    W8A8MatMul,
    NoisyW8A8MatMul,
)


def quantize_llama_model(
    model,
    weight_quant="per_channel",
    act_quant="per_token",
    quantize_bmm_input=False,
):

    from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP

    for name, m in model.model.named_modules():
        if isinstance(m, LlamaMLP):
            m.gate_proj = W8A8Linear.from_float(
                m.gate_proj, weight_quant=weight_quant, act_quant=act_quant
            )
            m.up_proj = W8A8Linear.from_float(
                m.up_proj, weight_quant=weight_quant, act_quant=act_quant
            )
            m.down_proj = W8A8Linear.from_float(
                m.down_proj, weight_quant=weight_quant, act_quant=act_quant
            )

        elif isinstance(m, LlamaAttention):
            m.q_proj = W8A8Linear.from_float(
                m.q_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.k_proj = W8A8Linear.from_float(
                m.k_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.v_proj = W8A8Linear.from_float(
                m.v_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.o_proj = W8A8Linear.from_float(
                m.o_proj, weight_quant=weight_quant, act_quant=act_quant
            )

    return model


def quantize_llama_model_error(
    model,
    weight_quant="per_channel",
    act_quant="per_token",
    quantize_bmm_input=True,
    err_prob=0,
):
   
    from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP

    i = 0
    for name, m in model.model.named_modules():
        if isinstance(m, LlamaMLP):
            if i == 0:
                m.gate_proj = NoisyW8A8Linear.from_float(
                    m.gate_proj,
                    weight_quant=weight_quant,
                    act_quant=act_quant,
                    quantize_output=quantize_bmm_input,
                    err_prob=err_prob,
                )
                m.up_proj = NoisyW8A8Linear.from_float(
                    m.up_proj,
                    weight_quant=weight_quant,
                    act_quant=act_quant,
                    quantize_output=quantize_bmm_input,
                    err_prob=err_prob,
                )
                m.down_proj = NoisyW8A8Linear.from_float(
                    m.down_proj,
                    weight_quant=weight_quant,
                    act_quant=act_quant,
                    quantize_output=quantize_bmm_input,
                    err_prob=err_prob,
                )
                i += 1
            else:
                m.gate_proj = W8A8Linear.from_float(
                    m.gate_proj,
                    weight_quant=weight_quant,
                    act_quant=act_quant,
                    quantize_output=quantize_bmm_input,
                )
                m.up_proj = W8A8Linear.from_float(
                    m.up_proj,
                    weight_quant=weight_quant,
                    act_quant=act_quant,
                    quantize_output=quantize_bmm_input,
                )
                m.down_proj = W8A8Linear.from_float(
                    m.down_proj,
                    weight_quant=weight_quant,
                    act_quant=act_quant,
                    quantize_output=quantize_bmm_input,
                )

        elif isinstance(m, LlamaAttention):
            if i == 0:
                m.q_proj = NoisyW8A8Linear.from_float(
                    m.q_proj,
                    weight_quant=weight_quant,
                    act_quant=act_quant,
                    quantize_output=quantize_bmm_input,
                    err_prob=err_prob,
                )
                m.k_proj = NoisyW8A8Linear.from_float(
                    m.k_proj,
                    weight_quant=weight_quant,
                    act_quant=act_quant,
                    quantize_output=quantize_bmm_input,
                    err_prob=err_prob,
                )
                m.v_proj = NoisyW8A8Linear.from_float(
                    m.v_proj,
                    weight_quant=weight_quant,
                    act_quant=act_quant,
                    quantize_output=quantize_bmm_input,
                    err_prob=err_prob,
                )
                m.o_proj = NoisyW8A8Linear.from_float(
                    m.o_proj,
                    weight_quant=weight_quant,
                    act_quant=act_quant,
                    err_prob=err_prob,
                )
                m.matmul1 = NoisyW8A8MatMul(
                    act_quant=act_quant, quantize_output=False
                )
                m.matmul2 = NoisyW8A8MatMul(
                    act_quant=act_quant, quantize_output=True
                )
            else:
                m.q_proj = W8A8Linear.from_float(
                    m.q_proj,
                    weight_quant=weight_quant,
                    act_quant=act_quant,
                    quantize_output=quantize_bmm_input,
                )
                m.k_proj = W8A8Linear.from_float(
                    m.k_proj,
                    weight_quant=weight_quant,
                    act_quant=act_quant,
                    quantize_output=quantize_bmm_input,
                )
                m.v_proj = W8A8Linear.from_float(
                    m.v_proj,
                    weight_quant=weight_quant,
                    act_quant=act_quant,
                    quantize_output=quantize_bmm_input,
                )
                m.o_proj = W8A8Linear.from_float(
                    m.o_proj, weight_quant=weight_quant, act_quant=act_quant
                )
                m.matmul1 = W8A8MatMul(
                    act_quant=act_quant, quantize_output=False
                )
                m.matmul2 = W8A8MatMul(
                    act_quant=act_quant, quantize_output=True
                )

    return model


class Evaluator:
 

    def __init__(self, dataset, tokenizer, device):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        def tokenize_function(examples):
            example = self.tokenizer(examples["text"])
            return example

        self.dataset = self.dataset.map(tokenize_function, batched=True)
        self.dataset.set_format(type="torch", columns=["input_ids"])

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        total, hit = 0, 0

        for batch in tqdm(self.dataset, desc="Evaluating"):
            input_ids = batch["input_ids"].to(self.device).unsqueeze(0)
            label = input_ids[:, -1]
            outputs = model(input_ids)
            last_token_logits = outputs.logits[:, -2, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()

        accuracy = hit / total
        return round(accuracy * 100, 3)


class Evaluator_ppl:
   

    def __init__(self, dataset, tokenizer, device, n_samples=40):
        self.tokenizer = tokenizer
        self.device = device
        self.n_samples = n_samples

        self.dataset = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
        self.dataset = self.dataset.input_ids.to(device)

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        nlls = []

        for i in tqdm(range(self.n_samples), desc="Evaluating..."):
            batch = self.dataset[:, (i * 2048) : ((i + 1) * 2048)]
            with torch.no_grad():
                lm_logits = model(batch).logits
            shift_logits = lm_logits[:, :-1, :].contiguous().float()
            shift_labels = self.dataset[
                :, (i * 2048) : ((i + 1) * 2048)
            ][:, 1:]

            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            neg_log_likelihood = loss.float() * 2048
            nlls.append(neg_log_likelihood)

        return torch.exp(torch.stack(nlls).sum() / (self.n_samples * 2048))


class Evaluator_x_sum:

    def __init__(self, dataset, tokenizer, device):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        self.prompt = (
            "For the following article, return a summary comprising of one sentence."
        )

        def tokenize_function(examples):
            example = self.tokenizer(f'{self.prompt}{examples["document"]}')
            return example

        self.dataset = self.dataset.map(tokenize_function)
        self.dataset.set_format(type="torch", columns=["input_ids"])
        self.summary = dataset["summary"]

    def evaluate(self, model, model_decode):
        model.eval()
        rouge1_sum_autoregressive = 0.0
        total = 0

        for i, example in enumerate(tqdm(self.dataset, desc="Evaluating")):
            input_ids = example["input_ids"].to(self.device).unsqueeze(0)
            input_token_len = input_ids.shape[1]

            num_tokens = 40
            top_k = 1
            top_p = 0.0
            summary_ids = autoregressive_sampling(
                x=input_ids,
                model=model,
                model_decode=model_decode,
                N=num_tokens,
                temperature=1,
                top_k=top_k,
                top_p=top_p,
            )
            summary_text = self.tokenizer.decode(
                summary_ids[0, input_token_len:],
                skip_special_tokens=True,
            )

            scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
            rouge_score_autoregressive = scorer.score(summary_text, self.summary[i])
            rouge1_sum_autoregressive += rouge_score_autoregressive["rouge1"].fmeasure
            total += 1

        return rouge1_sum_autoregressive / total


err_prob_list = [8, 16, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
ppl_normal_list = []
ppl_noisy_list = []
acc_noisy_list = []

start_time = time.time()

for i in range(len(err_prob_list)):
    start_time_i = time.time()
    err_prob = err_prob_list[i]
    print(err_prob)

    print("loading model")
    model_fp32_noisy = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        torch_dtype=torch.float32,
        device_map="auto",
    )
    act_scales = torch.load("act_scales/llama-2-7b.pt")

    print("smoothing")
    smooth_lm(model_fp32_noisy, act_scales, 0.5)

    print("tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    print("loading dataset")
    dataset_lambada = load_dataset("lambada", split="validation")
    evaluator = Evaluator(dataset_lambada, tokenizer, "cuda")

    n_samples = 40
    dataset_wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    evaluator_ppl = Evaluator_ppl(dataset_wikitext, tokenizer, "cuda", n_samples=n_samples)

    print("Inject_error...")
    noisy_model = quantize_llama_model_error(model_fp32_noisy, err_prob=err_prob)
    print("noisy_model_quantized")

    print("evaluating")
    acc_noisy = evaluator.evaluate(noisy_model)
    acc_noisy_list.append(acc_noisy)

    ppl_noisy = evaluator_ppl.evaluate(noisy_model)
    ppl_noisy_list.append(ppl_noisy.cpu().item())

    print("acc_noisy", acc_noisy, "pll_noisy", ppl_noisy)

    end_time_i = time.time()
    print("time_i", (end_time_i - start_time_i) / 60)

end_time = time.time()

print("acc_noisy_list", acc_noisy_list)
for item in acc_noisy_list:
    print(item)

print("ppl_noisy_list", ppl_noisy_list)
for items in ppl_noisy_list:
    print(items)

time_total = (end_time - start_time) / 60
print("time_sum,", time_total)
