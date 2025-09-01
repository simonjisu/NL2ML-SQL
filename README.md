# NL2ML-SQL

Natural language to ML-ready SQL for BigQuery ML and PostgresML, plus an intent router that classifies whether a query needs plain retrieval or machine learning. This repo includes:

- Data generation from public BigQuery datasets
- NLQ to intent extraction model (format usable for ML SQL)
- LLM pipelines to generate SQL templates and extract intent back from SQL
- PostgresML-based model selection experiments and figure scripts

Below are setup instructions, how to create data, and how to run the experiments.

**Contents**
- Setup
- Environment
- Data Creation
- Intent Training/Evaluation
- Experiments (Routing, Typical NL→SQL, ML Selection)
- Figures

## Setup

- Python: 3.11+
- GPU: recommended for vLLM and Unsloth fine-tuning/inference (CUDA 12.4 as pinned via PyTorch index)

Install dependencies with either uv (recommended) or pip:

- uv (reads `pyproject.toml` + `uv.lock`):
  - `uv sync`
  - `source .venv/bin/activate`

- pip:
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -e .`

## Environment

Copy `.env.sample` to `.env` and fill the keys you will use:

- `OPENAI_API_KEY`: for OpenAI LLMs
- `PROJECT_ID`: your Google Cloud project id
- `DATA_ID`: BigQuery public dataset project, e.g. `bigquery-public-data`
- `GOOGLE_CSE_ID`, `GOOGLE_API_KEY`: for LangChain Google Search (used in `exp_typical.py`)

BigQuery access uses Application Default Credentials. Either:

- Set `GOOGLE_APPLICATION_CREDENTIALS` to a service account JSON key
- Or run `gcloud auth application-default login`

## Data Creation

There are two main steps: (1) generate unlabeled intermediate intents from BigQuery schemas; (2) turn intents into explicit/implicit NL questions and augmented prompts.

1) Unlabeled intent pool from public BigQuery tables
- Script: `src/gen_unlabeled.py`
- Requires Google Cloud BigQuery access and `PROJECT_ID` in `.env`
- Example:
  - `python src/gen_unlabeled.py --output_file data/unlabeled_pool_1k.json --max_per_table 50 --target_pool_size 1000`

2) NL question generation from intents (explicit/implicit)
- Script: `src/gen_question.py`
- Uses OpenAI LLM for paraphrasing and BigQuery for realistic value perturbations
- Inputs: an intent file (`--input_file`), a column-alias mapping (`--mapping_file`), and a data dictionary (`--data_dict`)
- Example:
  - `python src/gen_question.py --input_file data/train_seed.jsonl --output_file data/train_dataset.jsonl --mapping_file data/data_mapping_cleaned.json --data_dict data/data_dictionaries.json`

Repo includes sample datasets under `data/` you can use directly:
- `data/train_seed_cots.jsonl` (seed with chain-of-thought)
- `data/train_dataset.jsonl`, `data/test_dataset.jsonl` (final train/test)
- Utility: `python data/get_dataset_stats.py` prints stats for seed/train/test splits

## Intent Training/Evaluation

We fine-tune a compact model with Unsloth, then evaluate with several inference strategies.

Train (SFT)
- Script: `intent_train.py`
- Example:
  - `python intent_train.py --model_name unsloth/Llama-3.1-8B-unsloth-bnb-4bit --max_seq_length 5500 --dataset_path ./data/train_dataset.jsonl --save_directory ./outputs`

Evaluate
- Script: `intent_eval.py`
- Methods: `cot` (chain-of-thought), `mv` (majority voting), `sc` (best-of-n via score)
- Example (CoT):
  - `python intent_eval.py --input_path ./data/test_dataset.jsonl --output_path ./exps/full_test.json --adapter_path ./outputs --model_name unsloth/Llama-3.1-8B-unsloth-bnb-4bit --method cot`
- Example (MV with n=3):
  - `python intent_eval.py --input_path ./data/test_dataset.jsonl --output_path ./exps/full_test_mv.json --adapter_path ./outputs --model_name unsloth/Llama-3.1-8B-unsloth-bnb-4bit --method mv --n 3 --temperature 0.8 --top_p 0.9`

Output JSON includes predictions and evaluation diagnostics via `src/evaluator.py`.

## Experiments

### Routing (IR vs ML)

Classify if a natural language question requires retrieval (SQL ops) or machine learning.

- Script: `exp_router.py`
- Input: a JSONL with fields `{"question": ..., "label": 0|1}`
- OpenAI modes:
  - `python exp_router.py --model gpt --openai_model_type gpt-4o-mini --n_inference 1 --output_path exps/router_gpt.json`
- vLLM mode (GPU):
  - `python exp_router.py --model llama --n_inference 3 --output_path exps/router_llama.json`

Results are written as JSON with `predicted_label` and token counts.

### Typical NL2SQL (Algorithm selection + SQL templates)

Run an LLM pipeline to pick an algorithm and generate BigQuery/Postgres SQL templates, then extract intent back from the SQL.

- Script: `exp_typical.py`
- Requires search keys in `.env` for `GoogleSearchAPIWrapper`
- Example (BigQuery, OpenAI):
  - `python exp_typical.py --input_path data/test_dataset.jsonl --output_path exps/typical_bigquery.jsonl --model_name gpt-4o-mini --platform bigquery`
- Example (Postgres, OpenAI):
  - `python exp_typical.py --input_path data/test_dataset.jsonl --output_path exps/typical_postgres.jsonl --model_name gpt-4o-mini --platform postgres`

Each output line contains the original question/schema, chosen algorithm, generated SQL, and extracted intent (via `src/extract_intent_bigqueryml.py` or `src/extract_intent_postgresml.py`).

### SQL Generation (TemplateGenerators)

Generate training and inference SQL using the TemplateGenerators on pre-computed records.

- Script: `src/gen_sqls.py`
- Input: JSONL/JSON where each record contains:
  - `dataset_name`, `table_name`
  - `schema` (data dictionary; `{table: {columns: {col: {type: ...}}}}`)
  - `intent` (`time_series`, `target_column`, `inference_condition`, `update_condition`, `task`)
  - `model_name` (algorithm name supported by the chosen platform)
- Output: JSONL; each line holds `train_output` and `inference_output`.

Usage (PostgresML):
- `python -m src.gen_sqls --input_path exps/dataset_for_templates.jsonl --output_path exps/gen_sqls_postgres.jsonl --platform postgres --auto_preprocess --test_size 0.1`

Usage (BigQuery ML):
- `export PROJECT_ID=your-gcp-project`
- `python -m src.gen_sqls --input_path exps/dataset_for_templates.jsonl --output_path exps/gen_sqls_bigquery.jsonl --platform bigquery`

Notes
- Table-specific column exclusions default to a built-in map; override via `--exclusions_json <file.json>` (map: `{table_name: [cols...]}`).
- BigQuery generator uses `PROJECT_ID` (required) and `DATA_ID` (default: `bigquery-public-data`).
- Output normalization:
  - Postgres generator returns dicts (view SQL, train SQL, etc.) under `payload`.
  - BigQuery generator returns raw SQL strings under `sql`.

Example input record (JSONL):
{"dataset_name":"thelook_ecommerce","table_name":"orders","schema":{"orders":{"columns":{"user_id":{"type":"INTEGER"},"num_of_item":{"type":"INTEGER"}}}},"intent":{"time_series":"False","target_column":"<col>num_of_item</col>","inference_condition":["<col>num_of_item</col><op>></op><val>1</val>"],"update_condition":[],"task":"regression"},"model_name":"xgboost"}

### ML Selection with PostgresML

Train multiple classical models per intent using PostgresML, collect metrics, and build a comparison table.

1) Start PostgresML (Docker)
- `docker run -it \
    --name postgresml \
    -v postgresml_data:/lab_shared/postgresqlml \
    -p 25432:5432 \
    -p 28000:8000 \
    ghcr.io/postgresml/postgresml:2.10.0 \
    sudo -u postgresml psql -d postgresml`

2) Load sample tables into the `public` schema
- CSVs are under `archived/dataset/*_cleaned.csv`
- Create tables named to match file stems without `_cleaned` (e.g., `orders`, `fips`)
- You can load via Pandas + SQLAlchemy, psql `\copy`, or your preferred tool
- Minimal Python snippet (run locally with access to the container):
  - `python - <<'PY'
import pandas as pd
from sqlalchemy import create_engine
engine = create_engine('postgresql+psycopg2://postgresml@localhost:25432/postgresml')
for p in [
  'archived/dataset/orders_cleaned.csv',
  'archived/dataset/outpatient_charges_2014_cleaned.csv',
  'archived/dataset/disclosures_cleaned.csv',
  'archived/dataset/fips_cleaned.csv',
  'archived/dataset/mbb_historical_teams_games_cleaned.csv',
  'archived/dataset/historical_runups_cleaned.csv',
  'archived/dataset/solar_potential_by_postal_code_cleaned.csv',
]:
    name = p.split('/')[-1].replace('_cleaned','').replace('.csv','')
    df = pd.read_csv(p)
    df.to_sql(name, engine, schema='public', if_exists='replace', index=False)
print('Done')
PY`

3) Run the experiment
- Script: `exp_mlselection.py` (uses `src/tempgenerator.PostgresTemplateGenerator`)
- It builds train SQL per (dataset, table, task, model), executes training, and collects metrics into CSV
- `python exp_mlselection.py`
- Output: `exps/exp_mlselection.csv`

Notes
- The engine URL is `postgresql+psycopg2://postgresml@localhost:25432/postgresml` (change if you map differently)
- The dataset→table map is hardcoded in `exp_mlselection.py`

## Figures

Use `draw_figures.py` to reproduce figures from experiment outputs.

- Skyline for ML selection: reads `exps/exp_mlselection.csv`
  - `python -c "import draw_figures as d; d.plot_mlselection()"`
  - Saves `figs/skyline_mlselection.pdf`

- Confusion matrices (example values):
  - `python -c "import draw_figures as d; d.plot_confusion_matrix(None,None)"`
  - Saves `figs/confusion_matrices.pdf`

- Intent detection summary (expects `exps/full_test.csv`):
  - `python -c "import draw_figures as d; d.plot_intent_detection()"`
  - Saves `figs/full_test.pdf`

## Tips & Troubleshooting

- BigQuery access: ensure ADC is set; test with `bq` CLI or `google-cloud-bigquery` quickstart
- OpenAI quota: question generation (`src/gen_question.py`) and `exp_typical.py` may incur costs
- GPU memory: vLLM and Unsloth may require >12GB VRAM; reduce batch sizes if OOM
- PostgresML datasets: verify tables exist and have expected columns before running `exp_mlselection.py`

## Project Layout

- `src/` core modules: prompts, intent extraction from SQL, template generators
- `data/` prepared datasets and helpers
- `exps/` experiment outputs (CSV/JSON)
- Top-level scripts: `exp_typical.py`, `exp_router.py`, `exp_mlselection.py`, `intent_train.py`, `intent_eval.py`, `draw_figures.py`
 - Utility: `src/gen_sqls.py` to batch-generate training/inference SQL from TemplateGenerators
