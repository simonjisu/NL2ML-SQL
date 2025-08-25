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

sampling_params = SamplingParams(
    n=1, 
    temperature=0.4, 
    top_p=0.95,
    max_tokens=2048,
    logprobs=1,
)

import os, json
from tqdm import tqdm

def get_batch(dataset, batch_size=8):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i:i + batch_size]

# ── 파일 경로 ────────────────────────────────────────────────────
dst_path = "/home/jsjang/code/NL2ML-SQL/SFT_zero_shot_v3.jsonl"
os.makedirs(os.path.dirname(dst_path), exist_ok=True)

batch_size = 8
results = []
with open(dst_path, "w") as fout:
    for batch in tqdm(get_batch(test_dataset, batch_size=batch_size), 
                      desc="Batch Processing", total=len(test_dataset)//batch_size):
        
        prompts  = [x["prompt"] for x in batch]
        ground_truths = [x["references"] for x in batch]
        req_outs: list[RequestOutput] = llm.generate(prompts, sampling_params)

        for out, gt in zip(req_outs, ground_truths):
            result: dict = parse_output(out)
            record = {
                "ground_truth": gt,
                "prediction": result['json_obj'],
                "total_tokens": result['total_token_cnt'],
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()                      # 중간에 끊겨도 안전