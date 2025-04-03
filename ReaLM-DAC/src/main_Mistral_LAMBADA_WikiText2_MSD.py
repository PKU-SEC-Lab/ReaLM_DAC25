import os
import re
import time
import torch
import numpy as np
import json
import pdb
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)
from transformers.models.mistral.modeling_mistral import MistralAttention, MistralMLP
from smoothquant.error_inject import W8A8Linear, NoisyW8A8Linear, W8A8MatMul, NoisyW8A8MatMul
from smoothquant.smooth import smooth_lm
from datasets import load_dataset
from tqdm import tqdm

def quantize_mistral_model(
    model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=False
):
    """
    Quantize Mistral model layers.
    """
    for name, m in model.model.named_modules():
        if isinstance(m, MistralMLP):
            m.gate_proj = W8A8Linear.from_float(
                m.gate_proj, weight_quant=weight_quant, act_quant=act_quant
            )
            m.up_proj = W8A8Linear.from_float(
                m.up_proj, weight_quant=weight_quant, act_quant=act_quant
            )
            m.down_proj = W8A8Linear.from_float(
                m.down_proj, weight_quant=weight_quant, act_quant=act_quant
            )
        elif isinstance(m, MistralAttention):
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
            m.matmul1 = W8A8MatMul(act_quant=act_quant, quantize_output=False)
            m.matmul2 = W8A8MatMul(act_quant=act_quant, quantize_output=True)
    return model

def quantize_mistral_model_error(
    model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=True, err_prob=0
):
    """
    Quantize Mistral model layers with error injection.
    """
    i = 0
    for name, m in model.model.named_modules():
        if isinstance(m, MistralMLP):
            if i == 0:
                m.gate_proj = NoisyW8A8Linear.from_float(
                    m.gate_proj, weight_quant=weight_quant, act_quant=act_quant,
                    quantize_output=quantize_bmm_input, err_prob=err_prob
                )
                m.up_proj = NoisyW8A8Linear.from_float(
                    m.up_proj, weight_quant=weight_quant, act_quant=act_quant,
                    quantize_output=quantize_bmm_input, err_prob=err_prob
                )
                m.down_proj = NoisyW8A8Linear.from_float(
                    m.down_proj, weight_quant=weight_quant, act_quant=act_quant,
                    quantize_output=quantize_bmm_input, err_prob=err_prob
                )
                i += 1
            else:
                m.gate_proj = W8A8Linear.from_float(
                    m.gate_proj, weight_quant=weight_quant, act_quant=act_quant,
                    quantize_output=quantize_bmm_input
                )
                m.up_proj = W8A8Linear.from_float(
                    m.up_proj, weight_quant=weight_quant, act_quant=act_quant,
                    quantize_output=quantize_bmm_input
                )
                m.down_proj = W8A8Linear.from_float(
                    m.down_proj, weight_quant=weight_quant, act_quant=act_quant,
                    quantize_output=quantize_bmm_input
                )
        elif isinstance(m, MistralAttention):
            if i == 0:
                m.q_proj = NoisyW8A8Linear.from_float(
                    m.q_proj, weight_quant=weight_quant, act_quant=act_quant,
                    quantize_output=quantize_bmm_input, err_prob=err_prob
                )
                m.k_proj = NoisyW8A8Linear.from_float(
                    m.k_proj, weight_quant=weight_quant, act_quant=act_quant,
                    quantize_output=quantize_bmm_input, err_prob=err_prob
                )
                m.v_proj = NoisyW8A8Linear.from_float(
                    m.v_proj, weight_quant=weight_quant, act_quant=act_quant,
                    quantize_output=quantize_bmm_input, err_prob=err_prob
                )
                m.o_proj = NoisyW8A8Linear.from_float(
                    m.o_proj, weight_quant=weight_quant, act_quant=act_quant,
                    err_prob=err_prob
                )
                m.matmul1 = NoisyW8A8MatMul(
                    act_quant=act_quant, quantize_output=False, err_prob=err_prob
                )
                m.matmul2 = NoisyW8A8MatMul(
                    act_quant=act_quant, quantize_output=True, err_prob=err_prob
                )
            else:
                m.q_proj = W8A8Linear.from_float(
                    m.q_proj, weight_quant=weight_quant, act_quant=act_quant,
                    quantize_output=quantize_bmm_input
                )
                m.k_proj = W8A8Linear.from_float(
                    m.k_proj, weight_quant=weight_quant, act_quant=act_quant,
                    quantize_output=quantize_bmm_input
                )
                m.v_proj = W8A8Linear.from_float(
                    m.v_proj, weight_quant=weight_quant, act_quant=act_quant,
                    quantize_output=quantize_bmm_input
                )
                m.o_proj = W8A8Linear.from_float(
                    m.o_proj, weight_quant=weight_quant, act_quant=act_quant
                )
                m.matmul1 = W8A8MatMul(act_quant=act_quant, quantize_output=False)
                m.matmul2 = W8A8MatMul(act_quant=act_quant, quantize_output=True)
    return model

class Evaluator:
    """
    Evaluator for accuracy of the model on the dataset.
    """
    def __init__(self, dataset, tokenizer, device):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device
        self.dataset = self.dataset.map(
            lambda examples: self.tokenizer(examples['text']), batched=True
        )
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
    Evaluator for perplexity (PPL) of the model.
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

err_prob_list = [1,2,4,8,16,32,64,128,256]  # Change to your desired list of error frequencies
ppl_normal_list = []
ppl_noisy_list = []
acc_noisy_list = []

start_time = time.time()

for err_prob in err_prob_list:
    start_time_i = time.time()
    print(f"Error Probability: {err_prob}")

    print("Loading model...")
    model_fp32_noisy = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-v0.1", torch_dtype=torch.float32, device_map="auto"
    )
    act_scales = torch.load("act_scales/Mistral-7B-v0.1.pt")

    smooth_lm(model_fp32_noisy, act_scales, 0.5)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

    print("Loading dataset...")
    dataset_lambada = load_dataset("lambada", split="validation")
    evaluator = Evaluator(dataset_lambada, tokenizer, "cuda")
    evaluator_ppl = Evaluator_ppl(load_dataset("wikitext", "wikitext-2-raw-v1", split="test"), tokenizer, "cuda", n_samples=40)

    print("Injecting error...")
    noisy_model = quantize_mistral_model_error(model_fp32_noisy, err_prob=err_prob)
    print("Noisy model quantized.")

    print("Evaluating...")
    acc_noisy = evaluator.evaluate(noisy_model)
    acc_noisy_list.append(acc_noisy)
    ppl_noisy = evaluator_ppl.evaluate(noisy_model)
    ppl_noisy_list.append(ppl_noisy.cpu().item())
    
    print(f"Accuracy: {acc_noisy}, Perplexity: {ppl_noisy}")

    end_time_i = time.time()
    print(f"Time for this iteration: {(end_time_i - start_time_i) / 60} minutes")

end_time = time.time()
print(f"Total time: {(end_time - start_time) / 60} minutes")

print(f"Accuracy list: {acc_noisy_list}")
print(f"Perplexity list: {ppl_noisy_list}")
