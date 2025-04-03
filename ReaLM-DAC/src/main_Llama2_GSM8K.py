import argparse
import gc
import os
import re
import time
from tqdm import tqdm

import numpy as np
import torch
import datasets
from datasets import load_dataset, load_from_disk
from sampling import autoregressive_sampling
from smoothquant.error_inject import (
    W8A8Linear,
    W8A8MatMul,
    NoisyW8A8MatMul,
)
from smoothquant.smooth import smooth_lm
import jsonlines
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Tokenizer,
    LlamaTokenizer,
    TextDataset,
)
from transformers.generation import GenerationConfig
from transformers.models.opt.modeling_opt import (
    OPTAttention,
    OPTDecoderLayer,
    OPTForCausalLM,
)


ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
ans_re1 = re.compile(r"(\-?[0-9][0-9\.\,]*)")
ans_re2 = re.compile(r'=\s*(\$?-?[0-9][0-9\.\,]*)')
prefix_sky1 = 'answer is'
prefix_sky2 = 'the answer is'
INVALID_ANS = "[invalid]"


def get_match_str(match, idx):
    match_str = match[idx]
    match_str = match_str.replace(",", "")
    if match_str.endswith('.'):
        match_str = match_str[:-1]
    if match_str.endswith('.00'):
        match_str = match_str[:-3]
    if match_str.endswith('.0'):
        match_str = match_str[:-2]
    return match_str


def doc_to_text(doc):
    return (
        fewshot_prompt 
        + "\nQuestion: " 
        + doc["question"] 
        + "\nLet's think step by step\n"
    )


def decode(tokens_list, tokenizer, raw_text_len):
    sents = []
    for tokens in tokens_list:
        tokens = tokens.cpu().numpy().tolist()
        sent = tokenizer.decode(tokens[raw_text_len:])
        sents.append(sent)
    return sents


def generate_sample(model, model_decode, tokenizer, input_txt):
    input_ids = tokenizer([input_txt], padding=False)["input_ids"]
    context_enc = torch.tensor(input_ids, device=model.device)
    input_token_len = len(input_ids[0])
    num_tokens = 150
    output_ids = autoregressive_sampling(
        x=context_enc,
        model=model,
        model_decode=model_decode,
        N=num_tokens,
        temperature=1,
        top_k=1,
        top_p=0.,
    )
    output_text = tokenizer.decode(
        output_ids[0, input_token_len:], 
        skip_special_tokens=True
    )
    return output_text


def extract_answer_hf(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return eval(match_str)
    return INVALID_ANS


def extract_answer(text):
    if prefix_sky1 in text:
        text = text.split(prefix_sky1)[-1]
    if prefix_sky2 in text:
        text = text.split(prefix_sky2)[-1]
    
    match1 = re.findall(ans_re1, text)
    match2 = re.findall(ans_re2, text)
    ans = []
    
    if match1:
        match_str1 = get_match_str(match1, -1)
        ans.append(match_str1)
    if match2:
        match_str2 = get_match_str(match2, -1).replace('$', '')
        ans.append(match_str2)
    
    return eval(ans[-1]) if ans else INVALID_ANS


def is_correct(completion, answer):
    completion = completion.split('</s>')[0]
    completion = completion.split('\n\n\n')[0]
    completion = completion.split("\n\n")[0]
    completion = completion.split("Question:")[0]

    gold = extract_answer_hf(answer)
    assert gold != INVALID_ANS, "No ground truth answer found in the document."
    
    try:
        clear_answer = extract_answer(completion)
    except Exception as error:
        print(f"Can't extract answer correctly: {error}")
        clear_answer = None
    
    return clear_answer == gold


def quantize_model(
    model, 
    weight_quant="per_channel", 
    act_quant="per_token", 
    quantize_bmm_input=False
):
    from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP
    from transformers.models.mistral.modeling_mistral import MistralAttention, MistralMLP

    for name, m in model.model.named_modules():
        if isinstance(m, (LlamaMLP, MistralMLP)):
            m.gate_proj = W8A8Linear.from_float(
                m.gate_proj, 
                weight_quant=weight_quant, 
                act_quant=act_quant
            )
            m.up_proj = W8A8Linear.from_float(
                m.up_proj,
                weight_quant=weight_quant,
                act_quant=act_quant
            )
            m.down_proj = W8A8Linear.from_float(
                m.down_proj,
                weight_quant=weight_quant,
                act_quant=act_quant
            )
        elif isinstance(m, (LlamaAttention, MistralAttention)):
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
                m.o_proj, 
                weight_quant=weight_quant, 
                act_quant=act_quant
            )
            m.matmul1 = W8A8MatMul(act_quant=act_quant, quantize_output=False)
            m.matmul2 = W8A8MatMul(act_quant=act_quant, quantize_output=True)
    return model


def quantize_error_model(
    model, 
    weight_quant="per_channel", 
    act_quant="per_token", 
    quantize_bmm_input=True, 
    err_prob=0
):
    from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP
    from transformers.models.mistral.modeling_mistral import MistralAttention, MistralMLP

    for name, m in model.model.named_modules():
        if isinstance(m, (LlamaMLP, MistralMLP)):
            m.gate_proj = W8A8Linear.from_float(
                m.gate_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input
            )
            m.up_proj = W8A8Linear.from_float(
                m.up_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input
            )
            m.down_proj = W8A8Linear.from_float(
                m.down_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input
            )
        elif isinstance(m, (LlamaAttention, MistralAttention)):
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
                m.o_proj, 
                weight_quant=weight_quant, 
                act_quant=act_quant
            )
            m.matmul1 = W8A8MatMul(act_quant=act_quant, quantize_output=False)
            m.matmul2 = NoisyW8A8MatMul(
                act_quant=act_quant, 
                quantize_output=True,
                err_prob=err_prob
            )
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HF checkpoint.")
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=str,
        help="Checkpoint path",
        default="",
    )
    parser.add_argument(
        "-f", 
        "--sample-input-file", 
        type=str, 
        default=None
    )
    parser.add_argument(
        "-o", 
        "--sample-output-file", 
        type=str, 
        default="Llama_gsm8k_res.jsonl"
    )
    args = parser.parse_args()

    fewshot_prompt = open("./gsm8k_prompt.txt").read()
    if args.sample_input_file:
        dataset = load_from_disk(args.sample_input_file)
    else:
        config = datasets.DownloadConfig(
            resume_download=True, 
            max_retries=100
        )
        dataset = load_dataset("gsm8k", "main", download_config=config)

    test = dataset["test"].select(range(300))
    err_prob_list = [0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    gsm8k_acc_list = []
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-hf", 
        trust_remote_code=True, 
        padding_side='left'
    )
    
    if "qwen" in "meta-llama/Llama-2-7b-hf".lower():
        tokenizer.pad_token = '<|extra_0|>'
        tokenizer.eos_token = '<|endoftext|>'
    else:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"

    start_time_all = time.time()
    for i, err_prob in enumerate(err_prob_list):
        print(f'Current error probability: {err_prob}')
        time_start = time.time()
        
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf", 
            device_map="auto", 
            trust_remote_code=True
        ).eval()
        
        act_scales = torch.load('act_scales/llama-2-7b.pt')
        smooth_lm(model, act_scales, 0.85)
        error_model = quantize_error_model(model, err_prob=err_prob)
        del model
        gc.collect()

        f_output = jsonlines.Writer(open(args.sample_output_file, "w", encoding="utf-8"))
        acc_res = []
        
        for doc in tqdm(test, desc='Evaluating'):
            context = doc_to_text(doc)
            completion = generate_sample(error_model, error_model, tokenizer, context)
            acc = is_correct(completion, doc["answer"])
            doc["completion"] = completion
            doc["acc"] = acc
            f_output.write(doc)
            acc_res.append(acc)
        
        f_output.close()
        gsm8k_acc = np.mean(acc_res)
        print(f"Accuracy: {gsm8k_acc:.4f}")
        gsm8k_acc_list.append(gsm8k_acc)
        
        error_model.cpu()
        del error_model
        torch.cuda.empty_cache()
        
        time_i = time.time() - time_start
        print(f'Time elapsed: {time_i/60:.2f} minutes')

    print("Final results:")
    for acc in gsm8k_acc_list:
        print(f"{acc:.4f}")
    
    total_time = (time.time() - start_time_all) / 60
    print(f'Total execution time: {total_time:.2f} minutes')