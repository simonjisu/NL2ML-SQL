from huggingface_hub import login

# Replace with your actual token (you can also load it from an environment variable for safety)
hf_token = "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"

login(token=hf_token)

from vllm import LLM, SamplingParams, RequestOutput
from pathlib import Path
# LLM setting
model_dir = str(Path("./Merged_Model").resolve())

llm = LLM(
    model=model_dir, 
    dtype="bfloat16", 
    trust_remote_code=True,
    seed=3074, 
    max_model_len=4096, 
    gpu_memory_utilization=0.8,
    enforce_eager=False,
)

import json 
import re

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


test_path = "./test_set_small.jsonl"
test_data = load_jsonl(test_path)
test_dataset = prepare_dataset(test_data)

PAT = re.compile(
    r"^###\s*Output:?\s*[\r\n]+(.*?)(?=^\s*###|\Z)",
    flags=re.I | re.S | re.M       #  I = case-insensitive, S = DOTALL, M = ^ at every line
)

def parse_output(out: RequestOutput):
    input_token_cnt = len(out.prompt_token_ids)  # input token count
    output_token_cnt = len(out.outputs[0].token_ids)  # output token count
    total_token_cnt = input_token_cnt + output_token_cnt  # total token count
    # some times pattern search doesn't work... 
    pred_str = out.outputs[0].text
    if PAT.search(pred_str):
        json_str = PAT.search(pred_str).group(1).strip()
    else:
        json_str = pred_str[-300:]

    try:
        json_obj = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"JSON decode error, will try `eval`: {e}")
        json_obj = eval(json_str)
    except Exception as e:
        print(f"Other error: {e}")
        json_obj = None

    return {
        'input_token_cnt': input_token_cnt,
        'output_token_cnt': output_token_cnt,
        'total_token_cnt': total_token_cnt,
        'json_obj': json_obj,
    }

# Best of N
N = 2
sampling_params = SamplingParams(
    n=1, 
    temperature=0.7, 
    top_p=0.95,
    max_tokens=2048,
    logprobs=1,
)

def best_of_n(outputs: list[RequestOutput]):
    results = []
    def score(out: RequestOutput):
        # mean log-probability per generated token
        return out.outputs[0].cumulative_logprob / max(len(out.outputs[0].token_ids), 1)

    for out in outputs:
        results.append(parse_output(out))
    best = max(outputs, key=score)
    return parse_output(best)['json_obj'], results

import os, json
from tqdm import tqdm

# ── 파일 경로 ────────────────────────────────────────────────────
src_path = "/home/jsjang/code/NL2ML-SQL/test_set_small.jsonl"
dst_path = "/home/jsjang/code/NL2ML-SQL/SFT_best_of_n(2).jsonl"
os.makedirs(os.path.dirname(dst_path), exist_ok=True)

with open(src_path, "r") as fin, open(dst_path, "w") as fout:
    for line, sample in tqdm(zip(fin, test_dataset), desc="Best-of-N", total=len(test_dataset)):
        # 1) 동일 프롬프트 N개 생성
        prompts  = [sample["prompt"]] * N
        req_outs = llm.generate(prompts, sampling_params)

        # 2) Best 후보와 전체 후보 파싱
        best_json, all_jsons = best_of_n(req_outs)

        # 3) all_jsons에서 total_tokens 빼기
        total_tokens = 0
        for candidate in all_jsons:
            total_tokens += candidate["total_token_cnt"]

        # 4) test_small에서 data 정보 읽기
        index = json.loads(line)
        nlq = index["instruction"]
        data_dict = index["input"]
        ground_truth = index["output"]

        # 5) 결과 레코드 구성
        record = {
            "instruction": nlq,                    # 옵션: 몇 번째 샘플인지
            "input": data_dict,
            "ground_truth": ground_truth,
            "best_of_n": best_json,
            "all_candidates": all_jsons,
            "total_tokens": total_tokens
        }

        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        fout.flush()                      # 중간에 끊겨도 안전