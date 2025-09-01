import os
import json
from typing import Any, Dict, List
from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm
from loguru import logger

# Import template generators
try:
    # When executed from the repo root with `python -m src.gen_sqls`
    from src.tempgenerator import BigQueryTemplateGenerator, PostgresTemplateGenerator
except Exception:  # pragma: no cover - fallback if run as a plain script
    # When executed as `python src/gen_sqls.py`
    from tempgenerator import BigQueryTemplateGenerator, PostgresTemplateGenerator

_ = load_dotenv(find_dotenv())  # read local .env if present

def _load_records(path: str) -> List[Dict[str, Any]]:
    """
    Load dataset records from a JSONL (one object per line) or JSON (array) file.
    Each record is expected to contain at least:
      - dataset_name
      - table_name
      - schema (data dictionary)
      - intent (with keys time_series, target_column, inference_condition, update_condition, task)
      - model_name (algorithm name that the generator supports)
    """
    if path.lower().endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                # Allow a wrapped object under a top-level key like {"data": [...]}
                for key in ("data", "records", "items"):
                    if key in data and isinstance(data[key], list):
                        return data[key]
                raise ValueError("JSON object provided but no array-like key ['data','records','items'] found.")
            if not isinstance(data, list):
                raise ValueError("Input JSON must be a list of records or a JSONL file.")
            return data


def _default_exclusions() -> Dict[str, List[str]]:
    """
    Default table-specific exclusion columns (mirrors what's used in exp_mlselection.py)
    """
    return {
        "orders": ["user_id", "order_id", "created_at", "returned_at", "shipped_at", "delivered_at"],
        "disclosures_13": ["record_id", "family_id", "blanket_scope", "disclosure_event", "pub_cleaned", "wg_name"],
        "outpatient_charges_2014": ["provider_id"],
        "fips": ["GeoName", "GeoFIPS"],
        "mbb_historical_teams_games": ["team_id", "name", "market", "opp_id", "opp_name", "opp_code", "opp_market"],
        "historical_runups": ["id", "tsevent_id", "location_name"],
        "solar_potential_by_postal_code": ["center_point", "install_size_kw_buckets"],
    }


def _wrap_output(obj: Any) -> Dict[str, Any]:
    """
    Normalize generator outputs into a common structure.
    - If the generator returns a dict (PostgresML), keep keys as-is under 'payload'.
    - If the generator returns a string (BigQuery), store under 'sql'.
    """
    if isinstance(obj, dict):
        return {"type": "dict", "payload": obj}
    else:
        return {"type": "sql", "sql": str(obj)}


def build_generator(platform: str):
    platform = platform.lower()
    if platform == "postgres":
        return PostgresTemplateGenerator()
    elif platform == "bigquery":
        # Requires env PROJECT_ID and DATA_ID
        project_id = os.getenv("PROJECT_ID")
        data_id = os.getenv("DATA_ID", "bigquery-public-data")
        if not project_id:
            raise EnvironmentError("PROJECT_ID environment variable is required for BigQuery generator.")
        return BigQueryTemplateGenerator(
            platform_type="bigquery",
            google_project_id=project_id,
            google_data_id=data_id,
        )
    else:
        raise ValueError(f"Unknown platform: {platform}")


def main(args=None):
    import argparse

    parser = argparse.ArgumentParser(description="Generate training/inference SQL from TemplateGenerators.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to JSON/JSONL with dataset records.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to write JSONL outputs.")
    parser.add_argument("--platform", type=str, choices=["postgres", "bigquery"], default="postgres")

    # Postgres-specific options
    parser.add_argument("--test_size", type=float, default=0.1, help="Evaluation split for PostgresML training.")
    parser.add_argument("--auto_preprocess", action="store_true", help="Auto-generate preprocess args for PostgresML.")

    # Optional external exclusions JSON (table -> [cols])
    parser.add_argument("--exclusions_json", type=str, default=None, help="Path to JSON mapping table_name -> columns to exclude.")

    ns = parser.parse_args(args)

    records = _load_records(ns.input_path)
    generator = build_generator(ns.platform)

    if ns.exclusions_json and os.path.exists(ns.exclusions_json):
        with open(ns.exclusions_json, "r", encoding="utf-8") as f:
            table_specific_exclusions = json.load(f)
    else:
        table_specific_exclusions = _default_exclusions()

    out_path = ns.output_path
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    written = 0
    errors: List[Dict[str, Any]] = []

    with open(out_path, "w", encoding="utf-8") as fout:
        for x in tqdm(records, total=len(records)):
            try:
                exclude_cols = table_specific_exclusions.get(x.get("table_name", ""), [])

                train_output = generator.gen(
                    dataset_name=x["dataset_name"],
                    table_name=x["table_name"],
                    schema=x["schema"],
                    intent=x["intent"],
                    is_train=True,
                    model_name=x["model_name"],
                    exclude_cols=exclude_cols,
                    test_size=ns.test_size,
                    auto_preprocess=ns.auto_preprocess,
                )

                inference_output = generator.gen(
                    dataset_name=x["dataset_name"],
                    table_name=x["table_name"],
                    schema=x["schema"],
                    intent=x["intent"],
                    is_train=False,
                    model_name=x["model_name"],
                    exclude_cols=exclude_cols,
                )

                rec_out = {
                    "dataset_name": x.get("dataset_name"),
                    "table_name": x.get("table_name"),
                    "model_name": x.get("model_name"),
                    "task": (x.get("intent", {}) or {}).get("task"),
                    "train_output": _wrap_output(train_output),
                    "inference_output": _wrap_output(inference_output),
                }
                fout.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
                written += 1
            except Exception as e:
                errors.append({
                    "record": x,
                    "error": str(e),
                })

    if errors:
        err_path = os.path.splitext(out_path)[0] + ".errors.json"
        with open(err_path, "w", encoding="utf-8") as ef:
            json.dump(errors, ef, indent=2, ensure_ascii=False)

    logger.info(f"Wrote {written} records to {out_path}. Errors: {len(errors)}")


if __name__ == "__main__":
    main()

