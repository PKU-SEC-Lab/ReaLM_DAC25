import os
import re
import time
import json
import torch
import numpy as np
import pdb
from tqdm import tqdm
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer
from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoderLayer, OPTForCausalLM
from smoothquant.smooth import smooth_lm
from smoothquant.error_inject import W8A8Linear, W8A8BMM, NoisyW8A8Linear, NoisyW8A8BMM
from datasets import load_dataset

def quantize_model(
    model, weight_quant='per_tensor', act_quant='per_tensor', quantize_bmm_input=True
):
    """
    Quantizes the model's layers by applying quantization to various modules like OPTDecoderLayer and OPTAttention.
    """
    for name, m in model.model.named_modules():
        if isinstance(m, OPTDecoderLayer):
            m.fc1 = W8A8Linear.from_float(m.fc1, weight_quant=weight_quant, act_quant=act_quant, quantize_output=True)
            m.fc2 = W8A8Linear.from_float(m.fc2, weight_quant=weight_quant, act_quant=act_quant)
        elif isinstance(m, OPTAttention):
            m.q_proj = W8A8Linear.from_float(
                m.q_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.k_proj = W8A8Linear.from_float(
                m.k_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.v_proj = W8A8Linear.from_float(
                m.v_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.out_proj = W8A8Linear.from_float(m.out_proj, weight_quant=weight_quant, act_quant=act_quant)

            m.bmm1 = W8A8BMM(act_quant=act_quant, quantize_output=False)
            m.bmm2 = W8A8BMM(act_quant=act_quant, quantize_output=True)
    return model

def quantize_model_error(
    model, weight_quant='per_tensor', act_quant='per_tensor', quantize_bmm_input=True, err_prob=0
):
    """
    Quantizes the model's layers with error injection.
    """
    for name, m in model.model.named_modules():
        if isinstance(m, OPTDecoderLayer):
            m.fc1 = W8A8Linear.from_float(m.fc1, weight_quant=weight_quant, act_quant=act_quant)
            m.fc2 = W8A8Linear.from_float(m.fc2, weight_quant=weight_quant, act_quant=act_quant)
        elif isinstance(m, OPTAttention):
            m.q_proj = W8A8Linear.from_float(
                m.q_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.k_proj = W8A8Linear.from_float(
                m.k_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.v_proj = W8A8Linear.from_float(
                m.v_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.out_proj = NoisyW8A8Linear.from_float(m.out_proj, weight_quant=weight_quant, act_quant=act_quant, err_prob=err_prob)

            m.bmm1 = W8A8BMM(act_quant=act_quant, quantize_output=False)
            m.bmm2 = W8A8BMM(act_quant=act_quant, quantize_output=True)
    return model

class Evaluator:
    """
    Evaluates the accuracy of the model on the given dataset.
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

# List of error frequencies
err_prob_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
ppl_normal_list = []
ppl_noisy_list = []
acc_noisy_list = []

start_time = time.time()

for err_prob in err_prob_list:
    start_time_i = time.time()
    print(f"Error Probability: {err_prob}")

    print("Loading model...")
    model_fp32_noisy = OPTForCausalLM.from_pretrained(
        'facebook/opt-1.3b', torch_dtype=torch.float32, device_map='auto'
    )

    act_scales = torch.load('act_scales/opt-1.3b.pt')
    smooth_lm(model_fp32_noisy, act_scales, 0.5)

    print("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('facebook/opt-1.3b')

    print("Loading dataset...")
    dataset_lambada = load_dataset('lambada', split='validation')
    evaluator = Evaluator(dataset_lambada, tokenizer, 'cuda')
    n_samples = 40
    dataset_wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    evaluator_ppl = Evaluator_ppl(dataset_wikitext, tokenizer, "cuda", n_samples=n_samples)

    print("Injecting error...")
    noisy_model = quantize_model_error(model_fp32_noisy, err_prob=err_prob)
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
