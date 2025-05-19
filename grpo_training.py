import re
from difflib import SequenceMatcher
import json
from unsloth import FastLanguageModel #, PatchFastRL
from trl import GRPOConfig, GRPOTrainer
import mlflow
from datetime import datetime as dt

user_prompt = """
We aim to extract structured machine learning configuration arguments and conditions from a natural language question and a given data dictionary.
These arguments are essential for automatically generating BigQuery ML SQL code.
The output must strictly follow the specified format and use the keys as described below.
The output must use column names from the Data Dictionary when using <col></col> tags in inference and update conditions.

### Output Format:
The output should be a JSON object containing the following keys:

1. **time_series** (boolean): Indicates whether the model is intended for time series forecasting.
   - Example: "time_series": "False"
   - Use "True" if the input involves time columns such as "Date", otherwise "False".

2. **target_column** (string): The column name that represents the target variable to predict.
   - Use the format: "<col>column_name</col>"
   - Example: "target_column": "<col>clarity</col>"
   - Make sure to use the same column names as in the Data Dictionary when target_column exist
   - Target_column can be an empty string when the task is "clustering"
   - Target_column can be an empty string or some column in Data Dictionary when the task is "anomaly_detection"

3. **inference_condition** (list of strings): A list of conditions used for inference or prediction. Each condition should specify a column, an operator, and a value.
   - Use the format: "<col>column_name</col><op>operator</op><val>value</val>"
   - Multiple conditions can be provided as a list.
   - Example: "inference_condition": ["<col>carat</col><op>>=</op><val>1.0</val>", "<col>color</col><op>=</op><val>J</val>"]

4. **update_condition** (list of strings, optional): A list of conditions for updating the data or model. Similar to `inference_condition`, it specifies a column, operator, and value.
   - Example: "update_condition": ["<col>color</col><op>=</op><val>G</val>"]
   - If there is no change in the conditions as per the instruction, this key should not be generated.
   - Make sure to use the same column names as in the the Data Dictionary.

5. **task** (string): The type of machine learning task to perform.
   - Common values: "classification", "regression", "clustering", "anomaly_detection"
   - Example: "task": "classification"

### Natural Language Question:
{instruction}

### Data Dictionary:
{data_dict}

"""

class RewardCalculator:
    def __init__(
        self,
        weights = None,
        use_fuzzy_matching = True,
        normalize = True,  # True means `filter` mode other than `grpo` mode
        verbose = False
    ):
        self.use_fuzzy_matching = use_fuzzy_matching
        self.verbose = verbose
        self.normalize = normalize
        self.fail_val = 0.0 if normalize else -1.0

        # Fixed weights
        self.weights = weights or {
            "time_series": 0.1,
            "target_column": 0.5,
            "inference_condition": 0.3,
            "update_condition": 0.4,
            "task": 0.7
        }

        # Required keys
        self.required_keys = ["time_series", "target_column", "inference_condition", "task"]


    def _match(self, a, b, key=None):

        def jaccard(set1, set2):
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            if union != 0:
                # Jaccard similarity
                result = intersection / union
                if result == 0.0:
                    return self.fail_val
                return result
            else:
                return self.fail_val

        def extract_condition_parts(condition_str):
            condition_str = condition_str.strip()

            col = re.findall(r"<col>(.*?)</col>", condition_str)
            op = re.findall(r"<op>(.*?)</op>", condition_str)
            val = re.findall(r"<val>(.*?)</val>", condition_str)

            col_val = col[0].strip() if col else ""
            op_val = op[0].strip() if op else ""
            val_val = val[0].strip() if val else ""

            # Step 2: Heuristics if any of the parts are missing
            if not (col_val and op_val and val_val):
                # Remove tags to simplify raw parsing
                clean_str = re.sub(r"</?[^>]+>", "", condition_str)

                # Try simple expression pattern: col op val
                match = re.match(r"([a-zA-Z0-9_.]+)\s*([=!<>]+)\s*(.+)", clean_str)
                if match:
                    if not col_val:
                        col_val = match.group(1).strip()
                    if not op_val:
                        op_val = match.group(2).strip()
                    if not val_val:
                        val_val = match.group(3).strip()

            return col_val, op_val, val_val

        def tag_completeness_score(cond_str, tags):
            present = sum(tag in cond_str for tag in tags)
            return present / len(tags)
        
        def score_pair(a_cond, b_cond):
            a_cond = a_cond.strip().lower()
            b_cond = b_cond.strip().lower()
            a_col, a_op, a_val = extract_condition_parts(a_cond)
            b_col, b_op, b_val = extract_condition_parts(b_cond)

            col_score = jaccard({a_col}, {b_col})
            op_score = jaccard({a_op}, {b_op})
            val_score = int(SequenceMatcher(None, a_val, b_val).ratio() >= 0.9)

            if col_score == 0 or op_score == 0 or val_score == 0:
                return self.fail_val

            avg_score = (col_score + op_score + val_score) / 3
            tag_score = tag_completeness_score(a_cond, ["<col>", "</col>", "<op>", "</op>", "<val>", "</val>"])
            if tag_score != 1.0:
                return self.fail_val

            return avg_score

        if key in {"inference_condition", "update_condition"}:
            a_list = a if isinstance(a, list) else [a]
            b_list = b if isinstance(b, list) else [b]

            if not a_list and not b_list:
                return 1.0
            if not a_list or not b_list:
                return self.fail_val

            used_b_indices = set()
            matched_scores = []

            for a_cond in a_list:
                best_score = self.fail_val
                best_j = None
                for j, b_cond in enumerate(b_list):
                    if j in used_b_indices:
                        continue
                    score = score_pair(a_cond, b_cond)
                    if score > best_score:
                        best_score = score
                        best_j = j
                if best_j is not None:
                    used_b_indices.add(best_j)
                matched_scores.append(best_score)  # score is 0.0 if unmatched

            # Final score is average over max(len(predicted), len(ground_truth))
            final_score = sum(matched_scores) / max(len(a_list), len(b_list))
            return final_score

        # Default Jaccard (for all non-condition fields)
        a_str = " ".join(map(str, a)) if isinstance(a, list) else str(a)
        b_str = " ".join(map(str, b)) if isinstance(b, list) else str(b)

        if not a_str.strip() and not b_str.strip():
            return 1.0
        if not a_str.strip() or not b_str.strip():
            return self.fail_val

        if key == "target_column":

            def strip_tags(text):
                return re.sub(r"</?[^>]+>", "", text).strip().lower()

            a_clean = strip_tags(a_str)
            b_clean = strip_tags(b_str)

            tag_score = tag_completeness_score(a_str, ["<col>", "</col>"])
            if tag_score != 1.0:
                return self.fail_val

            sim_score = jaccard({a_clean}, {b_clean})
            return sim_score

        a_tokens = set(a_str.lower().split())
        b_tokens = set(b_str.lower().split())

        return jaccard(a_tokens, b_tokens)


    def weighted_accuracy(self, predicted, ground_truth):
        """Computes weighted accuracy + diagnostics"""
        matches = {}
        diagnostics = {}

        weights = self.weights.copy()

        # Required keys
        for key in self.required_keys:
            sim_score = self._match(predicted.get(key, []), ground_truth.get(key, []), key=key)
            matches[key] = sim_score
            diagnostics[key] = weights.get(key, 0) * sim_score

        # Optional key: update_condition
        has_update_condition = "update_condition" in ground_truth and (ground_truth.get("update_condition") not in (None, []))
        if has_update_condition:
            if predicted.get("update_condition") is None:
                predicted['update_condition'] = []
            sim_score = self._match(
                predicted.get("update_condition", []),
                ground_truth.get("update_condition", []),
                key="update_condition"
            )
            matches["update_condition"] = sim_score
            diagnostics["update_condition"] = weights.get("update_condition") * sim_score

        # Dynamically decide which keys to include in normalization
        active_keys = self.required_keys.copy()
        if has_update_condition:
            active_keys.append("update_condition")
        
        if self.normalize:
            # filter mode: normalize the score
            max_possible_score = sum(weights.get(k, 0) for k in active_keys)
            weighted_score = sum(diagnostics.values())
            final_score = max(0.0, min(1.0, weighted_score / max_possible_score))
        else:
            # grpo mode: No normalization
            final_score = sum(diagnostics.values())

        if self.verbose:
            print("[Reward Diagnostics]")
            print("Matches:", matches)
            print("Diagnostics (per-key contribution):", diagnostics)
            print("Weighted Score:", final_score)

        return round(final_score, 6), matches, diagnostics
    
    def get_min_possible_reward(self, ground_truth, convert_to_max=False):
        w = sum([self.weights.get(k) for k in ground_truth.keys() if self.weights.get(k) is not None])
        r = round(self.fail_val * w, 6)
        return -r if convert_to_max else r
    
    def self_check(self, intermediate_output):
        """Checks for presence of required keys."""
        for key in self.required_keys:
            if key not in intermediate_output:
                return {"status": "INVALID", "reason": f"Missing required key: {key}"}
        return {"status": "VALID", "intermediate_output": intermediate_output}

class Evaluator:
    def __init__(self, mode: str = "filter", threshold: float = 0.9, verbose: bool = False):
        """
        Args:
            mode: 'filter' or 'grpo'
            threshold: filter cutoff (used only in filter mode)
        """
        assert mode in ("filter", "grpo")
        self.mode = mode
        self.threshold = threshold
        self.calculator = RewardCalculator(verbose=verbose, normalize=True if mode == "filter" else False)

    def evaluate(self, predicted: dict, ground_truth: dict) -> dict:
        validity = self.calculator.self_check(predicted)
        if validity["status"] != "VALID":
            result = {"score": 0.0, "diagnostics": {"status": validity["status"], "reason": validity["reason"]}}
            if self.mode == "filter":
                return {"keep": False, **result}
            elif self.mode == "grpo":
                return {"reward": self.calculator.get_min_possible_reward(ground_truth), **result}

        score, matches, diagnostics = self.calculator.weighted_accuracy(predicted, ground_truth)

        if self.mode == "filter":
            return {
                "keep": score >= self.threshold,
                "score": score,
                "diagnostics": {
                    "matches": matches,
                    "contributions": diagnostics,
                    "total": score
                }
            }
        elif self.mode == "grpo":
            return {
                "reward": score,
                "diagnostics": {
                    "matches": matches,
                    "contributions": diagnostics,
                    "total": score
                }
            }

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# Prepares the prompt and reference from loaded data
def prepare_dataset(data):
    dataset = []
    for ex in data:
        try:
            instruction = ex["instruction"]
            input_str = json.dumps(ex["input"], indent=2)
            output_obj = ex["output"]

            user_formatted = user_prompt.format(
                instruction=instruction,
                data_dict=input_str
            )

            dataset.append({
                "prompt": user_formatted,
                "references": output_obj
            })

        except Exception as e:
            print(f"Formatting error: {e}")
            continue

    return dataset


# def extract_json_block(text):
#     """
#     Extracts and parses the first JSON object from a string.
#     It assumes that the JSON block starts with '{' and attempts to find a balanced block.
#     """

# # find the position of the last open brace
#     start = text.rfind('{')
#     if start == -1:
#         return None

#     # now scan forward to find the matching closing brace
#     brace_count = 0
#     for i, c in enumerate(text[start:], start):
#         if c == '{':
#             brace_count += 1
#         elif c == '}':
#             brace_count -= 1
#             if brace_count == 0:
#                 candidate = text[start:i+1]
#                 try:
#                     return json.loads(candidate.replace('None','null'))
#                 except json.JSONDecodeError:
#                     return None

#     # if we never balanced off
#     return None

def correctness_reward_func(completions, references, **kwargs):
    """
    completions: list of generated outputs (strings)
    reference:   list of ground truth dicts (one per prompt)
    """
    rewards = []
    total_em = 0.0
    total_pm = 0.0
    PAT = re.compile(
        r"^###\s*Output:?\s*[\r\n]+(.*?)(?=^\s*###|\Z)",
        flags=re.I | re.S | re.M       #  I = case-insensitive, S = DOTALL, M = ^ at every line
    )
    for i, (pred_str, ref) in enumerate(zip(completions, references)):
        if PAT.search(pred_str):
            output_str = PAT.search(pred_str).group(1).strip()
        else:
            output_str = pred_str[-300:]

        if VERBOSE:
            print(f"[{i}-Pred] {output_str}")
            print(f"[{i}-Target] {ref}")
        try:
            pred = json.loads(output_str)
            evaluation_output = evaluator.evaluate(pred, ref)
            r = evaluation_output.get("reward")
            
            if VERBOSE:
                x = evaluator_filter.evaluate(pred, ref) 
                print(f"[{i}-Filter] {x['keep']}, {x['score']}")
                total_em += float(x.get("keep") == True)
                total_pm += x.get("score")

        except Exception:
            r = evaluator.calculator.get_min_possible_reward(ref)
        rewards.append(r)
    
    if VERBOSE:
        em = total_em / len(references)
        pm = total_pm / len(references)
        print(f'Scores: EM {em:.4f}, PM {pm:.4f} | Rewards: {", ".join([f"{r:.4f}" for r in rewards])}')

    return rewards

def format_reward_fn(completions, **kwargs):
    rewards = []
    for completion in completions:
        reward = 0.0
        if completion.count("### Chain of Thought") >= 1:
            reward += 0.5
            reward -= 0.5 * (completion.count("### Chain of Thought") - 1)
        if completion.count("### Output") >= 1:
            reward += 0.5
            reward -= 0.5 * (completion.count("### Output") - 1)
        rewards.append(reward)

    if VERBOSE:
        print(f"Format rewards: {', '.join([f'{r:.4f}' for r in rewards])}")
    return rewards

def length_reward_func(completions, **kwargs):
    rewards = []
    COT_PAT = re.compile(
        r"^###\s*Chain of Thought:?\s*[\r\n]+(.*?)(?=^\s*###|\Z)",
        flags = re.I | re.S | re.M
    )
    OUT_PAT = re.compile(
        r"^###\s*Output:?\s*[\r\n]+(.*?)(?=^\s*###|\Z)",
        flags = re.I | re.S | re.M
    )
    for completion in completions:
        reward = 0.0
        thought = COT_PAT.search(completion)
        output = OUT_PAT.search(completion)
        if thought:
            thought = thought.group(1).strip()
            n_thought_tokens = len(tokenizer.encode(thought))
            if n_thought_tokens > 100:
                reward += 0.25
            if n_thought_tokens < 500:
                reward += 0.25
            if n_thought_tokens > 600:
                reward -= n_thought_tokens * 0.001
        if output:
            output = output.group(1).strip()
            garbage = len(tokenizer.encode(completion.split(output)[1]))
            reward -= garbage * 0.001
        
        rewards.append(reward)
    if VERBOSE:
        print(f"length rewards: {', '.join([f'{r:.4f}' for r in rewards])}")
    return rewards

if __name__ == "__main__":
    mlflow.set_experiment("GRPO Training")

    VERBOSE = False
    max_seq_length = 4096
    dtype = None
    load_in_4bit = True
    lora_rank = 16
    # Load and prepare the dataset
    # train_path = "train_1_CoT_final.jsonl" 
    train_path = "./train_merged_CoT_final.jsonl"
    train_data = load_jsonl(train_path)
    train_dataset = prepare_dataset(train_data)

    evaluator = Evaluator(mode="grpo")
    evaluator_filter = Evaluator(mode="filter", threshold=1.0)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "./Seed_fine_tuned_llama_3.1_8B",
        max_seq_length = max_seq_length,
        load_in_4bit = load_in_4bit,
        dtype = dtype,
        max_lora_rank = lora_rank,
        fast_inference = True,
        gpu_memory_utilization = 0.6
    )
    # useful for reward functions
    # EOS_TOKEN = tokenizer.eos_token  
    max_prompt_length = 2200

    training_args = GRPOConfig(
        learning_rate = 2e-5,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.01,
        warmup_ratio = 0.1,
        warmup_steps = 0,
        lr_scheduler_type = "cosine",
        optim = "paged_adamw_8bit",
        logging_steps = 1,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 2, # Increase to 4 for smoother training
        num_generations = 4, # Decrease if out of memory
        max_prompt_length = max_prompt_length,
        max_completion_length = max_seq_length - max_prompt_length,
        num_train_epochs = 1, # Set to 1 for a full training run
        save_steps = 1,
        save_total_limit = 2,
        max_grad_norm = 0.1,
        report_to = "mlflow",
        run_name=f"GRPO-Llama-3.1B-{dt.now().strftime('%Y-%m-%d-%H-%M-%s')}",
        output_dir = "./grpo_outputs",
        vllm_max_model_len = max_seq_length,
        # use_vllm=True,
        # vllm_gpu_memory_utilization = 0.6, # Reduce if out of memory
    )

    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs=[
            correctness_reward_func,
            format_reward_fn,
            length_reward_func,
        ],
        args = training_args,
        train_dataset = train_dataset,
    )
    trainer.train(
        resume_from_checkpoint = False  if train_path == "train_1_CoT_final.jsonl" else True
    )

    import os
    number = train_path.split("_")[1]
    save_directory = f"./GRPO_fine_tuned_llama_3.1_8B-{number}"
    os.makedirs(save_directory, exist_ok=True)
    trainer.model.save_pretrained(save_directory)
    trainer.tokenizer.save_pretrained(save_directory)
    print(f"✅ GRPO 모델이 '{save_directory}'에 저장되었습니다!")


    """
    Num examples = 6,420 | Num Epochs = 3 | Total steps = 19,260
     
    """