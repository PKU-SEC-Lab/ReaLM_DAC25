import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
from smoothquant.fake_quant import NoisyW8A8Linear, W8A8Linear


MODEL_CACHE_DIR = "/opt/pretrained_models/llama-3-8b"
MODEL_NAME = "meta-llama/Meta-Llama-3-8B"  


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE_DIR)
tokenizer.pad_token = tokenizer.eos_token  


def quantize_llama_model_error(
    model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=True, err_prob = 0
):
    from transformers.models.llama.modeling_llama import (
        LlamaAttention,
        LlamaMLP,
    )
    i = 0
    for name, m in model.model.named_modules():
        if isinstance(m,LlamaMLP):
            if i == 30:
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
            # else:
            #     m.gate_proj = W8A8Linear.from_float(
            #         m.gate_proj, weight_quant=weight_quant, act_quant=act_quant,quantize_output=quantize_bmm_input
            #     )
            #     m.up_proj = W8A8Linear.from_float(
            #         m.up_proj, weight_quant=weight_quant, act_quant=act_quant,quantize_output=quantize_bmm_input
            #     )
            #     m.down_proj = W8A8Linear.from_float(
            #         m.down_proj, weight_quant=weight_quant, act_quant=act_quant,quantize_output=quantize_bmm_input
            #     )
        elif isinstance(m,LlamaAttention):
            if i == 30:
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
            # else:
            #     m.q_proj = W8A8Linear.from_float(
            #         m.q_proj,
            #         weight_quant=weight_quant,
            #         act_quant=act_quant,
            #         quantize_output=quantize_bmm_input,
            #     )

            #     m.k_proj = W8A8Linear.from_float(
            #         m.k_proj,
            #         weight_quant=weight_quant,
            #         act_quant=act_quant,
            #         quantize_output=quantize_bmm_input,
            #     )

            #     m.v_proj = W8A8Linear.from_float(
            #         m.v_proj,
            #         weight_quant=weight_quant,
            #         act_quant=act_quant,
            #         quantize_output=quantize_bmm_input,
            #     )

            #     m.o_proj = W8A8Linear.from_float(
            #         m.o_proj, weight_quant=weight_quant, act_quant=act_quant
            #     )                
            i=i+1
            print(i)
    return model

def render_example(example):
    """
    Given the example as a dictionary, render it as three torch tensors:
    - tokens (the tokens of context + completion, of size 4xN, as there are always 4 candidates)
    - mask (is 1 in the region of the candidate completion, where we evaluate likelihoods)
    - label (the index of the correct completion, which we hope has the highest likelihood)
    """
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    # Encode context and endings
    ctx_tokens = tokenizer.encode(ctx, add_special_tokens=False)
    tok_rows = []
    mask_rows = []
    for end in endings:
        end_tokens = tokenizer.encode(" " + end, add_special_tokens=False)
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0]*len(ctx_tokens) + [1]*len(end_tokens))

    # Padding
    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    return tokens, mask, label

@torch.no_grad()
def evaluate(model, dataset, device):
    model.eval()

    num_correct_norm = 0
    num_correct = 0
    num_total = 0
    for example in tqdm(dataset, desc="Evaluating"):
        tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)

        # Get the logits
        logits = model(tokens).logits
        # Evaluate the autoregressive loss at all positions
        shift_logits = logits[..., :-1, :].contiguous()
        shift_tokens = tokens[..., 1:].contiguous()
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_shift_tokens = shift_tokens.view(-1)
        shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
        shift_losses = shift_losses.view(tokens.size(0), -1)

        # Calculate loss for completion region
        shift_mask = mask[..., 1:].contiguous()
        masked_shift_losses = shift_losses * shift_mask
        sum_loss = masked_shift_losses.sum(dim=1)
        avg_loss = sum_loss / shift_mask.sum(dim=1)

        # Choose the option with the lowest loss
        pred = sum_loss.argmin().item()
        pred_norm = avg_loss.argmin().item()


        # Update stats
        num_total += 1
        num_correct += int(pred == int(label))
        num_correct_norm += int(pred_norm == int(label))
    
    return num_correct, num_correct_norm


  
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", type=str, default="cuda", help="the device to use")
    parser.add_argument("-m", "--MSD", type=str, default="0", help="MSD")
    args = parser.parse_args()


    err_freq=[1e-8, 1e-7,1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    for i in range(len(err_freq)):
        print(err_freq[i])
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE_DIR, torch_dtype=torch.float32).to("cuda")
        print('quantizing')
        noisy_model=quantize_llama_model_error(model, err_prob=err_freq[i])
     
        dataset = load_dataset("hellaswag", split="validation")

        print(noisy_model)
        answer=evaluate(model, dataset, args.device)
        print(err_freq[i], answer)

        del model, noisy_model
        torch.cuda.empty_cache()
