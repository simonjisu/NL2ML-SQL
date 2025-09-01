import os
import re
import json
import string
import random
import difflib
import datetime

from loguru import logger
from collections import OrderedDict
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
from google.cloud import bigquery
from dateutil.parser import parse as parsedate
from dotenv import find_dotenv, load_dotenv

import prompts
_ = load_dotenv(find_dotenv())  # read local .env file

DATA_ID = os.getenv("DATA_ID", "bigquery-public-data")
GOOGLE_PROJECT_ID = os.getenv("PROJECT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

column_value_cache = {}

def perturb_date(s):
    """
    If the input matches a known date format, parse it and perturb it by a few days.
    Otherwise, return the input unchanged.
    """
    if not s or not isinstance(s, str):
        return s

    s = s.strip()

    # Known formats mapped to strftime
    formats = [
        (r"^\d{2}-\d{2}$", "%m-%d"),  # MM-DD
        (r"^\d{8}$", "%Y%m%d"),  # YYYYMMDD
        (r"^\d{4}-\d{2}-\d{2}$", "%Y-%m-%d"),  # Date
        (r"^\d{2}/\d{2}/\d{4}$", "%m/%d/%Y"),  # MM/DD/YYYY
        (r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$", "%Y-%m-%d %H:%M:%S"),  # Datetime
        (r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+$", "%Y-%m-%d %H:%M:%S.%f"),  # +ms
        (r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+[+-]\d{2}:\d{2}$", "%Y-%m-%d %H:%M:%S.%f%z"),  # ms + offset
        (r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$", "%Y-%m-%d %H:%M:%S%z"),  # offset
        (r"^\d{4}-\d{2}-\d{2}T.*$", None),  # ISO fallback
        (r"^\d{2}-[A-Za-z]{3}-\d{2}( \d{2}:\d{2} [AP]M)?$", "%d-%b-%y %I:%M %p"),  # 02-JAN-13 12:00 AM
    ]

    try:
        for pattern, fmt in formats:
            if re.match(pattern, s):
                date_obj = parsedate(s)
                delta = datetime.timedelta(days = random.choice([-3, -2, -1, 1, 2, 3]))
                new_date = date_obj + delta

                if fmt is None:
                    return new_date.isoformat()
                elif fmt == "%d-%b-%y %I:%M %p":
                  result = new_date.strftime(fmt)

                  # Ensure actual perturbation (date_obj ≠ new_date)
                  if new_date.date() == parsedate(s).date():
                      new_date += datetime.timedelta(days=1)
                      result = new_date.strftime(fmt)

                  # Preserve uppercase month if original was like "DEC"
                  if s[3:6].isupper():
                      result = result[:3] + result[3:6].upper() + result[6:]

                  return result

                elif "%f" in fmt:
                    result = new_date.strftime(fmt)
                    return result[:-3] if ".%f" in fmt else result
                else:
                   if "%z" in fmt:
                     return new_date.isoformat(sep=" ", timespec="seconds")
                   return new_date.strftime(fmt)

        # Special UTC suffix case
        if "UTC" in s.upper():
            date_obj = parsedate(s)
            delta = datetime.timedelta(days=random.randint(-3, 3))
            new_date = date_obj + delta
            return new_date.strftime('%Y-%m-%dT%H:%M:%S') + " UTC"

    except Exception:
        pass

    return s  # return unchanged if not matched


def perturb_boolean(s):
    val = s.strip()
    lower_val = val.lower()

    mapping = {
        "true": "false",
        "false": "true",
        "yes": "no",
        "no": "yes",
        "1": "0",
        "0": "1",
        "y": "n",
        "n": "y",
        '*': "blank"
    }

    if lower_val not in mapping:
        return s  # unrecognized boolean form

    new_val = mapping[lower_val]

    # Preserve original casing
    if val.isupper():
        return new_val.upper()
    elif val[0].isupper():
        return new_val.capitalize()
    else:
        return new_val


def perturb_geopoint(s):
    try:
        s = s.strip()
        original_upper = s.upper().startswith("POINT")

        # Extract numeric values
        coords = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", s)]
        if len(coords) != 2:
            return s  # invalid format

        perturbed = [round(c + random.choice([
                  random.uniform(-2, -1e-2),
                  random.uniform(1e-2, 2)
              ]), 6) for c in coords]

        if original_upper:
            return f"POINT ({perturbed[0]} {perturbed[1]})"
        else:
            return f"({perturbed[0]}, {perturbed[1]})"

    except Exception:
        return s

# Fallback function for string mutation
def mutate_string(val_str):
    """Fallback mutation logic if no suitable BigQuery value is found."""
    if not val_str:
        return val_str

    val_str = val_str.strip()

    # Flip a digit if any are present
    digit_indices = [i for i, c in enumerate(val_str) if c.isdigit()]
    if digit_indices:
        idx = random.choice(digit_indices)
        old = val_str[idx]
        new = random.choice([d for d in string.digits if d != old])
        return val_str[:idx] + new + val_str[idx+1:]

    # Flip a letter if any are present
    alpha_indices = [i for i, c in enumerate(val_str) if c.isalpha()]
    if alpha_indices:
        idx = random.choice(alpha_indices)
        old = val_str[idx]
        new = random.choice([a for a in string.ascii_letters if a.lower() != old.lower()])
        return val_str[:idx] + new + val_str[idx+1:]

def perturb_string(
    val_str,
    client,
    project_id,
    dataset_id,
    table_id,
    column_name,
    similarity_threshold=0.9
):
    """
    Fetch a realistic but *meaningfully different* value from BigQuery.
    Avoid values that are too similar to the original.
    """
    if not val_str or not val_str.strip():
        return val_str

    val_str = val_str.strip()
    cache_key = f"{project_id}.{dataset_id}.{table_id}.{column_name}"

    # Fetch and cache values from BigQuery
    if cache_key not in column_value_cache:
        query = f"""
            SELECT DISTINCT {column_name}
            FROM `{project_id}.{dataset_id}.{table_id}`
            WHERE {column_name} IS NOT NULL
            LIMIT 50
        """
        try:
            query_job = client.query(query)
            all_values = [
                str(row[column_name]).strip()
                for row in query_job
                if row[column_name] is not None and str(row[column_name]).strip()
            ]
            column_value_cache[cache_key] = all_values
        except Exception as e:
            logger.error(f"[Error] BigQuery query failed: {e}")
            return val_str

    # Filter out similar values
    candidates = []
    for v in column_value_cache[cache_key]:
        if v.lower() == val_str.lower():
            continue
        similarity = difflib.SequenceMatcher(None, val_str, v).ratio()
        if similarity < similarity_threshold:
            candidates.append(v)

    # Choose or fallback
    if candidates:
        return random.choice(candidates)
    else:
        # Fallback: basic mutation
        logger.warning(f"[Fallback] No suitable value found for '{val_str}', applying synthetic mutation.")
        return mutate_string(val_str)


def perturb_digit(val_str):
    """Perturb a string by interpreting and transforming it."""
    # Try numeric
    try:
      number = float(val_str)
      if val_str.isdigit():
        original = int(number)
        for _ in range(3):  # Try up to 3 times
            delta = random.choice([-2, -1, 1, 2])
            new_val = original + delta
            if new_val > 0 and new_val != original:
                return str(new_val)
      else:
          factor = random.uniform(0.9, 1.1)
          new_val = number * factor
          return str(round(new_val, 2))
    except ValueError:
        pass

def perturb_value(val_str, col_type, client, project_id, dataset_id, table_id, column_name):
    if not val_str:
      return val_str

    val_str = val_str.strip()

    # Rule 1: Use declared boolean type
    if col_type == "BOOL":
        return perturb_boolean(val_str)

    # Rule 3: Use declared geography type
    if col_type == "GEOGRAPHY":
        return perturb_geopoint(val_str)

    # Rule 4: Trying date
    val_str_date = perturb_date(val_str)
    if val_str_date != val_str:
        return val_str_date

    # Rule 5: Try numeric (if not covered by col_type)
    if col_type in ["INTEGER", "FLOAT"]:
      try:
          float(val_str)
          return perturb_digit(val_str)
      except ValueError:
          pass

    # Rule 6: Fallback to string mutation
    return perturb_string(str(val_str), client, project_id, dataset_id, table_id, column_name )

def perturb_inference_conditions(data, client, project_id, dataset_id, table_id, type_map = None):
    """
    Perturb <val>...</val> tokens inside the inference_condition array.
    Adds a new field output['update_condition'].
    """

    perturbed_data = json.loads(json.dumps(data))  # deep copy
    output = perturbed_data.get("output", {})
    original_conditions = output.get("inference_condition", [])

    num_to_perturb = random.randint(1, len(original_conditions))
    indices_to_perturb = set(random.sample(range(len(original_conditions)), num_to_perturb))

    # Apply perturbation only to the selected subset
    perturbed_conditions = []
    for idx, cond in enumerate(original_conditions):
        if idx in indices_to_perturb:
            col_match = re.search(r"<col>(.*?)</col>", cond)
            col_name = col_match.group(1) if col_match else None
            col_type = type_map.get(col_name).get("type") if type_map and col_name else None

            new_cond = re.sub(
                r"<val>(.*?)</val>",
                lambda m: f"<val>{perturb_value(m.group(1), col_type, client, project_id, dataset_id, table_id, col_name)}</val>",
                cond
            )
            perturbed_conditions.append(new_cond)

    # Rebuild output dict with update_condition in desired position
    reordered_output = OrderedDict()
    for key, value in output.items():
        reordered_output[key] = value
        if key == "inference_condition":
            reordered_output["update_condition"] = perturbed_conditions

    perturbed_data["output"] = reordered_output
    return perturbed_data, list(indices_to_perturb)

def generate_explicit_question(intermediate_output, idx = None, perturbed = False):
    """
    Generate an explicit natural language question using structured tags and task-aware templates.
    """

    puzzle_pieces = {
        'regression': {
            'prefix': [
                "Predict", "Estimate", "Model", "Forecast", "Project", "Anticipate", "Quantify", "Simulate"
            ],
            'target': [
                "the future values of {target_column}",
                "expected trends in {target_column}",
                "changes over time in {target_column}",
                "how {target_column} will evolve",
                "upcoming fluctuations in {target_column}"
            ]
        },

        'classification': {
            'prefix': [
                "Classify", "Predict", "Determine", "Identify", "Assign", "Recognize", "Infer"
            ],
            'target': [
                "the correct category for {target_column}",
                "the most likely label of {target_column}",
                "the classification of {target_column}",
                "which group {target_column} belongs to",
                "the appropriate outcome for {target_column}"
            ]
        },

        'anomaly_detection': {
            'prefix': [
                "Detect", "Identify", "Predict", "Flag", "Spot", "Isolate"
            ],
            'target': [
                "anomalies in {target_column}",
                "irregular patterns in {target_column}",
                "unusual behavior in {target_column}",
                "outliers related to {target_column}",
                "deviations from normal in {target_column}"
            ]
        },

        'clustering': {
            'prefix': [
                "Group", "Cluster", "Organize", "Segment", "Predict", "Determine"
            ],
            'target': [
                "natural clusters within the data",
                "similar patterns across records",
                "groupings across input features",
                "subsets based on shared characteristics",
                "how data points are related",
                "inherent structure in the dataset"
        ],
        }
    }

    output = intermediate_output.get('output', {})
    time_series = output.get('time_series', 'False').lower() == 'true'
    target_column = output.get('target_column', '').strip() or "the data"
    inference_conditions = output.get('inference_condition', [])
    task = output.get('task', {})

    pattern = r'<col>(.*?)</col><op>(.*?)</op><val>(.*?)</val>'
    condition_clause = ""

    if perturbed:
        # Compare original and perturbed conditions
        update_conditions = output.get('update_condition', [])
        deltas = []

        perturb_map = dict(zip(sorted(idx), update_conditions))

        for i, orig in enumerate(inference_conditions):
            updated_cond = perturb_map.get(i)

            m1 = re.match(pattern, orig)
            m2 = re.match(pattern, updated_cond) if updated_cond else None

            if m1 and m2:
                col1, op1, val1 = m1.groups()
                col2, op2, val2 = m2.groups()
                if col1 == col2 and op1 == op2 and val1 != val2:
                  if op1 in [">", "<", ">=", "<="]:
                    deltas.append(
                        f"<col>{col1}</col> changes from <op>{op1}</op> <val>{val1}</val> to <op>{op2}</op> <val>{val2}</val>"
                    )
                  else:
                    deltas.append(
                        f"<col>{col1}</col> changes from <val>{val1}</val> to <val>{val2}</val>"
                    )
                else:
                    deltas.append(f"<col>{col1}</col> {op1} <val>{val1}</val>")
            elif m1:
                col1, op1, val1 = m1.groups()
                deltas.append(f"<col>{col1}</col> {op1} <val>{val1}</val>")

        if deltas:
            condition_clause = "if " + " and ".join(deltas)

    else:
        # Standard case: template-based explicit question
        parsed_conditions = []
        for cond in inference_conditions[:3]:
            match = re.match(pattern, cond)
            if match:
                col, op, val = match.groups()
                parsed_conditions.append(f"<col>{col}</col> {op} <val>{val}</val>")

        if parsed_conditions:
            condition_clause = "given " + " and ".join(parsed_conditions)

    # Choose NL puzzle pieces
    prefix = random.choice(puzzle_pieces.get(task, {}).get('prefix', ["Analyze"]))
    target = random.choice(puzzle_pieces.get(task, {}).get('target', [f"{target_column}"])).format(target_column=target_column)

    # Optional time series clause
    ts_clause = "over time" if time_series else ""

    # Assemble the final question
    parts = [f"{prefix} {target}"]
    if ts_clause:
        parts.append(ts_clause)
    if condition_clause:
        parts.append(condition_clause)

    question = ", ".join(parts) + "."

    return question

def data_map(explicit_question, table_id, mapping_dict):
    """
    Replace each <col>column</col> tag with a randomly sampled alias
    based on mapping_dict for a given table_id.
    """
    if table_id not in mapping_dict:
        return explicit_question  # fallback

    table_mapping = mapping_dict[table_id]
    pattern = r"<col>(.*?)</col>"

    def substitute_column(match):
        col_name = match.group(1)
        if col_name in table_mapping:
           alias = random.choice(table_mapping[col_name])
           return f"<col>{alias}</col>"
        return f"<col>{col_name}</col>"  # fallback

    return re.sub(pattern, substitute_column, explicit_question)

def generate_implicit_question(mapped_question):
    """
    Call LLM to rephrase an explicit question into an implicit conversational question.
    Removes structured tags like <col>, <val>, <op>.
    """
    prompt = prompts.GEN_IMPLICIT_QUESTION.format(mapped_question=mapped_question)
    try:
        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-4o-mini")
        total_cost = 0.0
        # Track cost
        with get_openai_callback() as cb:
            response = llm.invoke(prompt)
            total_cost = cb.total_cost
            prompt_tokens = cb.prompt_tokens
            completion_tokens = cb.completion_tokens
            logger.info(f"[Token usage] Prompt: {prompt_tokens}, Completion: {completion_tokens}, Cost: ${total_cost:.6f}")

        return response.content, total_cost, prompt_tokens, completion_tokens
    except Exception as e:
        return f"Error generating implicit question: {str(e)}"


def main(args):
    client = bigquery.Client(project=GOOGLE_PROJECT_ID)

    with open(args.input_file, "r") as f:
        data_list = json.load(f)

    with open(args.mapping_file, "r") as f:
        data_mapping = json.load(f)

    with open(args.data_dict, "r") as f:
        data_dict = json.load(f)

    results = []
    global_cost = 0.0
    total_prompt_tokens = 0.0
    total_completion_tokens = 0.0
    print_every = 100

    for i, data in enumerate(data_list):
        perturbed = False
        if random.random() < 0.6 and data.get("output", {}).get("inference_condition"):
            table_id = data.get("table_id")
            dataset_id = data.get("dataset_name")
            type_map = {}
            for entry in data_dict:
                tables = entry.get("tables", {})
                if table_id in tables:
                    type_map = tables[table_id].get("columns", {})
                    break
            data, idx = perturb_inference_conditions(data, client, DATA_ID, dataset_id, table_id, type_map)
            perturbed = True

        explicit_questions = []
        implicit_questions = []
        mapping = []

        for _ in range(5):  # 5 explicit questions
            try:
                if perturbed:
                    explicit = generate_explicit_question(data, idx, perturbed)
                else:
                    explicit = generate_explicit_question(data, perturbed)
            except Exception as e:
                  explicit = f"Error generating explicit question: {e}"

            explicit_questions.append(explicit)
            table_id = data.get("table_id", "")

            for _ in range(2):  # 2 implicit per explicit
                mapped_explicit = data_map(explicit, table_id, data_mapping)
                try:
                    implicit, cost, prompt_tokens, com_tokens = generate_implicit_question(mapped_explicit)
                    global_cost += cost
                    total_prompt_tokens += prompt_tokens
                    total_completion_tokens += com_tokens
                except Exception as e:
                    implicit = f"Error generating implicit question: {e}"
                mapping.append(mapped_explicit)
                implicit_questions.append(implicit)

        results.append({
            "original_data": data,
            "explicit_question": explicit_questions,
            "mapping": mapping,
            "implicit_question": implicit_questions,
            "type_2": perturbed
        })

        # Log progress
        if (i + 1) % print_every == 0:
            logger.info(f"[Processed {i + 1}/{len(data_list)}")


    # Final save
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Finished. Saved {len(results)} results to {args.output_file}.")
    logger.info(f"Total cost is {global_cost} for a total of {total_prompt_tokens} prompt tokens\
    and {total_completion_tokens} completion tokens.")


if __name__ == "__main__":
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--input_file", type=str, default="input.json", help="Input JSON file path")
        parser.add_argument("--output_file", type=str, default="output.json", help="Output JSON file path")
        parser.add_argument("--mapping_file", type=str, default="data_mapping_cleaned.json", help="Column mapping JSON file path")
        parser.add_argument("--data_dict", type=str, default="data_dictionaries.jsonl", help="Data dictionary JSON file path")
        args = parser.parse_args()
        main(args)