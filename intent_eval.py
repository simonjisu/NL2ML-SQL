import os
import re
import ast
import json
import torch
import torch.nn.functional as F

from typing import Optional
from tqdm import tqdm
from loguru import logger
from unsloth import FastLanguageModel
from src import prompts
from src.evaluator import Evaluator
from collections import Counter

PAT = re.compile(
    r"^\s*###\s*Output:?\s*[\r\n]+(.*?)(?=^\s*###|\Z)",
    flags=re.I | re.S | re.M,
)

def extract_output_block(pred_str: str) -> Optional[str]:
    if PAT.search(pred_str):
        json_str = PAT.search(pred_str).group(1).strip()
    else:
        json_str = pred_str[-300:]

    try:
        json_obj = json.loads(json_str)
    except json.JSONDecodeError as e:
        try:
            print(f"JSON decode error, will try `eval`: {e}")
            json_obj = eval(json_str)
        except Exception as e:
            print(f"Other error: {e}")
            json_obj = None

    return json_obj

def atomic_write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def load_unsloth_model(
        model_name:str, 
        adapter_path:str, 
        max_seq_length:int=5500, 
        load_in_4bit:bool=True, 
        dtype:None=None
    ):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        dtype=dtype,
    )
    model: FastLanguageModel = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    model.load_adapter(adapter_path, adapter_name="default")

    return model, tokenizer

def make_prompt(question: str, schema: str) -> str:
    prompt = prompts.INTENT_INSTRUCTION.format(question, schema)
    return prompt

def infer_cot(prompt, model, tokenizer, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2700,
            return_dict_in_generate=True,
            output_scores=False
        )
    decoded = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    output = extract_output_block(decoded)
    output = output if output is not None else decoded.strip()
    total_tokens = outputs.sequences[0].shape[-1]
    return output, total_tokens

def infer_mv(prompt, model, tokenizer, device, n, temperature, top_p):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2700,
            do_sample=True,
            num_return_sequences=n,
            temperature=temperature,
            top_p=top_p,
            return_dict_in_generate=True,
            output_scores=False
        )
    seqs = outputs.sequences
    samples = []
    for seq in seqs:
        decoded = tokenizer.decode(seq, skip_special_tokens=True)
        output  = extract_output_block(decoded)
        samples.append(output if output is not None else decoded.strip())

    best_output, _ = Counter(samples).most_common(1)[0]
    total_full_tokens = sum(seq.shape[-1] for seq in seqs)

    return best_output, total_full_tokens, samples

def infer_bon(prompt, model, tokenizer, device, n, temperature, top_p):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    # Generate N samples with scores
    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=2700,
            do_sample=True,
            num_return_sequences=n,
            temperature=temperature,
            top_p=top_p,
            return_dict_in_generate=True,
            output_scores=True,
        )

    # gen.sequences: [batch*n, seq_len]
    # gen.scores: tuple of length T (new tokens), each tensor [batch*n, vocab_size]
    seqs = gen.sequences  # shape: (n, input_len + T_max)
    scores_per_step = gen.scores  # tuple(len=T_max) of [n, V]
    total_full_tokens = int(sum(seq.shape[-1] for seq in seqs))

    # Decode all samples using extractor
    samples = []
    raw_texts = []
    for seq in seqs:
        decoded = tokenizer.decode(seq, skip_special_tokens=True)
        raw_texts.append(decoded)
        out = extract_output_block(decoded)
        samples.append(out if out is not None else decoded.strip())

    # Compute mean log-prob over generated tokens for each sequence
    # Align chosen token at step t with scores[t][j]
    # Note: scores are already post temperature/top_p processors in HF.
    n_seqs = seqs.size(0)
    T_max = len(scores_per_step)

    # Build a [n_seqs, T_max] tensor of generated token ids (pad past EOS)
    # Generated tokens = seqs[:, input_len: input_len+T_max]
    gen_token_ids = seqs[:, input_len:input_len + T_max]
    if gen_token_ids.size(1) < T_max:
        # Rare edge case if generation stopped early (< max_new_tokens)
        # Right-pad to T_max for easy indexing
        pad_cols = T_max - gen_token_ids.size(1)
        gen_token_ids = torch.nn.functional.pad(gen_token_ids, (0, pad_cols), value=pad_id)

    # For each step, get log-probs for the chosen tokens
    # Collect logp_step[j, t] = log P(token_jt | history)
    logp_step = []
    for t, step_scores in enumerate(scores_per_step):
        # step_scores: [n_seqs, vocab]; convert to log-probs
        log_probs = F.log_softmax(step_scores, dim=-1)  # [n_seqs, V]
        chosen = gen_token_ids[:, t].unsqueeze(-1)      # [n_seqs, 1]
        step_logp = log_probs.gather(dim=-1, index=chosen).squeeze(-1)  # [n_seqs]
        logp_step.append(step_logp)
    logp_step = torch.stack(logp_step, dim=1)  # [n_seqs, T_max]

    # Mask out everything after the first EOS for each sequence
    # Find first EOS index in the generated region; if none, use full length actually produced.
    # Actual produced length per seq = (seq_len - input_len) up to T_max
    produced_lens = (seqs.ne(pad_id).sum(dim=1) - input_len).clamp(min=0, max=T_max)  # [n_seqs]
    # Also cut at EOS if appears in generated slice
    eos_pos = torch.full((n_seqs,), fill_value=T_max, dtype=torch.long, device=seqs.device)
    if eos_id is not None:
        gen_slice = gen_token_ids  # [n_seqs, T_max]
        # locate first eos per row
        is_eos = (gen_slice == eos_id)
        any_eos = is_eos.any(dim=1)
        # Where EOS exists, argmax on flipped cumulative trick to get first True index
        # Simpler: use torch.where on argmax of is_eos.float() with a mask
        first_eos_idx = torch.argmax(is_eos.int(), dim=1)  # gives 0 if none, so guard with any_eos
        eos_pos = torch.where(any_eos, first_eos_idx, eos_pos)

    effective_lens = torch.minimum(produced_lens, eos_pos)  # [n_seqs]

    # Build mask [n_seqs, T_max] with 1 up to effective_len (exclusive), else 0
    arange_t = torch.arange(T_max, device=seqs.device).unsqueeze(0)  # [1, T_max]
    mask = (arange_t < effective_lens.unsqueeze(1)).float()          # [n_seqs, T_max]

    # Sum log-probs and divide by count (mean log-prob per generated token)
    sum_logp = (logp_step * mask).sum(dim=1)                  # [n_seqs]
    counts = mask.sum(dim=1).clamp(min=1.0)                   # avoid div by zero
    mean_logp = sum_logp / counts                             # [n_seqs]

    # Pick best index by highest mean log-prob
    best_idx = int(torch.argmax(mean_logp).item())
    best_output = samples[best_idx]

    return best_output, total_full_tokens, samples

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_unsloth_model(
        model_name=args.model_name,
        adapter_path=args.adapter_path,
        max_seq_length=args.max_seq_length
    )
    infer_function = {
        "cot": infer_cot, "mv": infer_mv, "sc": infer_bon
    }
    if args.method == "cot":
        additional_kwargs = {}
    elif args.method == "mv" or args.method == "sc":
        additional_kwargs = {
            "n": args.n,
            "temperature": args.temperature,
            "top_p": args.top_p
        }
    else:
        raise KeyError(f"Unknown inference method: {args.method}")

    infer_function = infer_function.get(args.method, None)
    if infer_function is None:
        raise KeyError(f"Unknown inference method: {args.method}")
    
    evaluator = Evaluator()

    if os.path.exists(args.output_path):
        with open(args.output_path, "r", encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = []

    done_count = len(results)
    logger.info(f"Loaded {done_count} existing results.")
    
    logger.info("Loading Data...")
    with open(args.input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line_number, line in enumerate(
        tqdm(lines[done_count:], desc="Evaluating"),
        start=done_count+1
    ):
        data = json.loads(line)

        question = data["question"]
        schema = data["schema"]
        ground_truth = data["intent"]

        if isinstance(ground_truth, str):
            try:
                ground_truth = json.loads(ground_truth)
            except json.JSONDecodeError:
                ground_truth = {}

        prompt = make_prompt(question, schema)
        generated_output, total_tokens, *_ = infer_function(
            prompt, model, tokenizer, device, **additional_kwargs
        )

        try:
            evaluate_prediction = evaluator.evaluate(generated_output, ground_truth)
        except Exception:
            evaluate_prediction = {"error": "evaluation_failed"}

        result_rec = {
            "question": question,
            "schema": schema,
            "ground_truth": ground_truth,
            "prediction": generated_output,
            "evaluate_prediction": evaluate_prediction,
            "total_tokens": total_tokens
        }

    results.append(result_rec)
    if (line_number % args.save_every) == 0:
        atomic_write_json(args.output_path, results)
        logger.info(f"Intermediate results saved to {args.output_path}")

    atomic_write_json(args.output_path, results)
    logger.info(f"Final results saved to {args.output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run inference on a JSONL file.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--adapter_path", type=str, default="./SFT_fine_tuned_llama_3.1_8B", required=True, help="Path to the adapter.")
    parser.add_argument("--model_name", type=str, default="unsloth/Llama-3.1-8B-unsloth-bnb-4bit", help="Model name.")
    parser.add_argument("--save_every", type=int, default=100, help="Save results every n examples.")
    parser.add_argument("--method", type=str, default="cot", help="CoT(cot) / Majority Voting(mv) / Best of N(bon).")
    parser.add_argument("--n", type=int, default=2, help="Number of samples for Majority Voting or Best of N.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling.")
    args = parser.parse_args()
    # input_path = "./test_set_final.jsonl"
    # output_path = "./exps/test_set_with_inference.jsonl"

    main(args)