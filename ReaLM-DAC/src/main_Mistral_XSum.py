import torch
import numpy as np
import re
import json
import os

from torch import nn
from functools import partial
from smoothquant.error_inject import W8A8Linear,NoisyW8A8Linear,W8A8MatMul,NoisyW8A8MatMul
from datasets import load_dataset
from smoothquant.smooth import smooth_lm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation import GenerationConfig

from torch.utils.data import DataLoader
from transformers import TextDataset, DataCollatorForLanguageModeling
from torch.nn import CrossEntropyLoss
import pdb
from tqdm import tqdm
import time

from sampling.autoregressive_sampling import autoregressive_sampling
import contexttimer
from rouge import Rouge
from rouge_score import rouge_scorer

def quantize_mistral_model(
    model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=False
):
    from transformers.models.llama.modeling_llama import (
        LlamaAttention,
        LlamaMLP,
    )

    from transformers.models.mistral.modeling_mistral import (
        MistralAttention,
        MistralMLP,
    )

    for name, m in model.model.named_modules():
        if isinstance(m, (LlamaMLP, MistralMLP)):
            m.gate_proj = W8A8Linear.from_float(
                m.gate_proj, weight_quant=weight_quant, act_quant=act_quant
            )
            m.up_proj = W8A8Linear.from_float(
                m.up_proj, weight_quant=weight_quant, act_quant=act_quant
            )
            m.down_proj = W8A8Linear.from_float(
                m.down_proj, weight_quant=weight_quant, act_quant=act_quant
            )
        elif isinstance(m, (LlamaAttention, MistralAttention)):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
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
            m.matmul1 = W8A8MatMul(act_quant=act_quant,quantize_output=False)
            m.matmul2 = W8A8Linear(act_quant=act_quant,quantize_output=True)   
    return model

def quantize_mistral_model_error(
    model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=True, err_prob = 0
):
    from transformers.models.mistral.modeling_mistral import (
        MistralAttention,
        MistralMLP,
    )
    i = 0
    for name, m in model.model.named_modules():
        #print(name)
        if isinstance(m,MistralMLP):
            if i==0:
                # m.gate_proj = W8A8Linear.from_float(
                #     m.gate_proj, weight_quant=weight_quant, act_quant=act_quant,quantize_output=quantize_bmm_input
                # )
                # m.up_proj = W8A8Linear.from_float(
                #     m.up_proj, weight_quant=weight_quant, act_quant=act_quant,quantize_output=quantize_bmm_input
                # )
                # m.down_proj = W8A8Linear.from_float(
                #     m.down_proj, weight_quant=weight_quant, act_quant=act_quant,quantize_output=quantize_bmm_input
                # )
                m.gate_proj = NoisyW8A8Linear.from_float(
                    m.gate_proj, weight_quant=weight_quant, act_quant=act_quant,quantize_output=quantize_bmm_input,err_prob=err_prob
                )
                m.up_proj = NoisyW8A8Linear.from_float(
                    m.up_proj, weight_quant=weight_quant, act_quant=act_quant,quantize_output=quantize_bmm_input,err_prob=err_prob
                )
                m.down_proj = NoisyW8A8Linear.from_float(
                    m.down_proj, weight_quant=weight_quant, act_quant=act_quant,quantize_output=quantize_bmm_input,err_prob=err_prob
                )
                i += 1
            else:
                m.gate_proj = W8A8Linear.from_float(
                    m.gate_proj, weight_quant=weight_quant, act_quant=act_quant,quantize_output=quantize_bmm_input
                )
                m.up_proj = W8A8Linear.from_float(
                    m.up_proj, weight_quant=weight_quant, act_quant=act_quant,quantize_output=quantize_bmm_input
                )
                m.down_proj = W8A8Linear.from_float(
                    m.down_proj, weight_quant=weight_quant, act_quant=act_quant,quantize_output=quantize_bmm_input
                )                
        elif isinstance(m, MistralAttention):
            if i==0:
                # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
                # m.q_proj = W8A8Linear.from_float(
                #     m.q_proj,
                #     weight_quant=weight_quant,
                #     act_quant=act_quant,
                #     quantize_output=quantize_bmm_input,
                # )
                m.q_proj = NoisyW8A8Linear.from_float(
                    m.q_proj,
                    weight_quant=weight_quant,
                    act_quant=act_quant,
                    quantize_output=quantize_bmm_input,
                    err_prob=err_prob
                )
                # m.k_proj = W8A8Linear.from_float(
                #     m.k_proj,
                #     weight_quant=weight_quant,
                #     act_quant=act_quant,
                #     quantize_output=quantize_bmm_input,
                # )
                m.k_proj = NoisyW8A8Linear.from_float(
                    m.k_proj,
                    weight_quant=weight_quant,
                    act_quant=act_quant,
                    quantize_output=quantize_bmm_input,
                    err_prob=err_prob
                )
                # m.v_proj = W8A8Linear.from_float(
                #     m.v_proj,
                #     weight_quant=weight_quant,
                #     act_quant=act_quant,
                #     quantize_output=quantize_bmm_input,
                # )
                m.v_proj = NoisyW8A8Linear.from_float(
                    m.v_proj,
                    weight_quant=weight_quant,
                    act_quant=act_quant,
                    quantize_output=quantize_bmm_input,
                    err_prob=err_prob
                )
                # m.o_proj = W8A8Linear.from_float(
                #     m.o_proj, weight_quant=weight_quant, act_quant=act_quant
                # )
                m.o_proj = NoisyW8A8Linear.from_float(
                    m.o_proj, weight_quant=weight_quant, act_quant=act_quant,err_prob=err_prob
                )
                # m.matmul1 = W8A8MatMul(act_quant=act_quant,quantize_output=False)
                m.matmul1 = NoisyW8A8MatMul(act_quant=act_quant,quantize_output=False, err_prob=err_prob)
                # m.matmul2 = W8A8Linear(act_quant=act_quant,quantize_output=True)
                m.matmul2 = NoisyW8A8MatMul(act_quant=act_quant,quantize_output=True, err_prob=err_prob)
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
                m.matmul1 = W8A8MatMul(act_quant=act_quant,quantize_output=False)
                m.matmul2 = W8A8Linear(act_quant=act_quant,quantize_output=True)                                 
    return model

class Evaluator:
    def __init__(self, dataset, tokenizer, device):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device
        # tokenize the dataset
        def tokenize_function(examples):
            example = self.tokenizer(examples['text'])
            return example

        self.dataset = self.dataset.map(tokenize_function, batched=True)
        self.dataset.set_format(type='torch', columns=['input_ids'])

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        for batch in tqdm(self.dataset, desc="Evaluating"):
            #pdb.set_trace()
            input_ids = batch['input_ids'].to(self.device).unsqueeze(0)
            label = input_ids[:, -1]
            outputs = model(input_ids)
            #pdb.set_trace()
            last_token_logits = outputs.logits[:, -2, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
        accuracy = hit / total
        acc = round(accuracy*100,3)
        return acc
   
class Evaluator_ppl:
    def __init__(self, dataset, tokenizer, device, n_samples=40):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        self.dataset = tokenizer(
            "\n\n".join(dataset["text"]), return_tensors="pt"
        ).input_ids.to(device)

        self.n_samples = n_samples

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        nlls = []
        for i in tqdm(range(self.n_samples), desc="Evaluating..."):
            batch = self.dataset[:, (i * 2048) : ((i + 1) * 2048)].to(model.device)
            with torch.no_grad():
                #pdb.set_trace()
                lm_logits = model(batch).logits
            shift_logits = lm_logits[:, :-1, :].contiguous().float()
            shift_labels = self.dataset[:, (i * 2048) : ((i + 1) * 2048)][:, 1:]
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            neg_log_likelihood = loss.float() * 2048
            nlls.append(neg_log_likelihood)
            # pdb.set_trace()

        return torch.exp(torch.stack(nlls).sum() / (self.n_samples * 2048))

class Evaluator_x_sum:
    def __init__(self, dataset, tokenizer, device, prompt):
        self.dataset=dataset
        self.tokenizer=tokenizer
        self.device=device
        # self.prompt = 'Please summarize the following article. '
        self.prompt=prompt

        def tokenize_function(examples):
            example=self.tokenizer(f'{self.prompt}\nDocument:{examples["document"]}\nSummarize the main content of the above article in one sentence:')
            return example
        
        self.dataset=self.dataset.map(tokenize_function)
        self.dataset.set_format(type='torch', columns=['input_ids'])
        self.summary=dataset['summary']

    def evaluate(self, model, model_decode):
        model.eval()
        rouge1_sum_autoregressive=0.
        total=0
        for i, example in enumerate(tqdm(self.dataset, desc='Evaluating')):
            input_ids=example['input_ids'].to(self.device).unsqueeze(0)
            input_token_len=input_ids.shape[1]
            # pdb.set_trace()
            num_tokens=40
            top_k = 1
            top_p = 0.
            summary_ids=autoregressive_sampling(x=input_ids,model=model,model_decode=model_decode, N=num_tokens,temperature=1, top_k=top_k, top_p=top_p)

            summary_text=tokenizer.decode(summary_ids[0,input_token_len:],skip_special_tokens=True)
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            rouge_score_autoregressive=scorer.score(summary_text, self.summary[i])
            # pdb.set_trace()
            rouge1_sum_autoregressive=rouge1_sum_autoregressive+rouge_score_autoregressive['rouge1'].fmeasure
            total=total+1

        return rouge1_sum_autoregressive/total

if __name__=='__main__':

    err_prob_list=[0.0, 1e-8,1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    ppl_normal_list=[]
    ppl_noisy_list=[]
    acc_noisy_list=[]
    x_sum__noisy_list=[]

    start_time = time.time()
    for i in range(len(err_prob_list)):
        start_time_i = time.time()
        err_prob=err_prob_list[i]
        print(err_prob)

        print('loading model')
        model_fp16_normal = AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-v0.1', torch_dtype=torch.float16, device_map='auto')
  
        act_scales = torch.load('act_scales/Mistral-7B-v0.1.pt')

        
        print('tokenizer')
        tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1')


        print('loading dataset EdinburghNLP/xsum')
        dataset_x_sum=load_dataset('EdinburghNLP/xsum',split='validation[:500]')


        with open('./xsum_prompt.txt','r',encoding='utf-8') as file:
            few_prompt = file.read()
        assert isinstance(few_prompt,str), "The document is not a string."
        evaluator_x_sum = Evaluator_x_sum(dataset_x_sum,tokenizer,'cuda', few_prompt)


        print('evaluating')
        x_sum = evaluator_x_sum.evaluate(model_fp16_normal,model_fp16_normal)
        print("x_sum", x_sum)
        x_sum__noisy_list.append(x_sum)

        end_time_i = time.time()
        print('time_i',(end_time_i - start_time_i)/60)

    end_time = time.time()
    print('x_sum_list',x_sum__noisy_list)
    time = (end_time-start_time)/60
    print('time_sum,',time)