from tempgenerator import BigQueryTemplateGenerator, PostgresTemplateGenerator
from tqdm import tqdm
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from loguru import logger

DATASET_TABLE_MAP = {
        "noaa_tsunami": "historical_runups",
        "thelook_ecommerce": "orders",
        "sunroof_solar": "solar_potential_by_postal_code",
        "sdoh_bea_cainc30": "fips",
        "medicare": "outpatient_charges_2014",
        "patents_dsep": "disclosures_13",
        "ncaa_basketball": "mbb_historical_teams_games"
    }
ENGINE_URL = "postgresql+psycopg2://{db_user}@{db_url}/{db_name}".format(
    db_user="postgresml",
    db_url="localhost:25432",
    db_name="postgresml"
)
PGML_SCHEMA = "public"
CSV_DIR = Path("./archived/dataset")
ENGINE = create_engine(ENGINE_URL, future=True)

def execute_query(query: str, params: dict = None, fetch: bool = False):
    """
    Execute a raw SQL query with SQLAlchemy.
    
    Args:
        query (str): The SQL query string (use :param for placeholders).
        params (dict): Dictionary of parameters to bind.
        fetch (bool): Whether to fetch results (for SELECT queries).
    
    Returns:
        list of dicts if fetch=True, else number of affected rows.

    Usage:
    ```
    # SELECT query
    rows = execute_query("SELECT * FROM users WHERE age > :age", {"age": 21}, fetch=True)
    print(rows)

    # INSERT query
    inserted = execute_query(
        "INSERT INTO users (name, age) VALUES (:name, :age)",
        {"name": "Alice", "age": 25}
    )
    print(f"Inserted rows: {inserted}")

    # UPDATE query
    updated = execute_query(
        "UPDATE users SET age = :age WHERE name = :name",
        {"name": "Alice", "age": 26}
    )
    print(f"Updated rows: {updated}")
    ```
    """
    
    try:
        with ENGINE.connect() as connection:
            result = connection.execute(text(query), params or {})
            
            if fetch:
                # Fetch results as list of dicts
                rows = [dict(row._mapping) for row in result]
                return rows
            else:
                # Commit changes for write queries
                connection.commit()
                return result.rowcount
    except SQLAlchemyError as e:
        logger.error(f"Database error: {e}")

with open('./data/train_mlselection.jsonl', 'r') as file:
    data = [json.loads(line) for line in file]

gen_class = PostgresTemplateGenerator
TABLE_DATASET_MAP = {v: k for k, v in DATASET_TABLE_MAP.items()}
dataset = []
checked_intents = set()
for d in tqdm(data, total=len(data)):
    intent = {k: v for k, v in d['intent'].items() 
              if k in ('time_series', 'target_column', 'task')}

    k = json.dumps(intent)
    if k in checked_intents:
        continue
    original_schema = d['schema']['tables']
    assert len(list(original_schema.keys())) == 1, "len table is not 1"
    table_name = list(original_schema.keys())[0]
    dataset_name = TABLE_DATASET_MAP[table_name]
    query = f"""
        SELECT COLUMN_NAME, DATA_TYPE
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = 'public'
        AND TABLE_NAME = '{table_name}';"""
    schema = execute_query(query, fetch=True)
    schema = {
        table_name: {
            'columns': {col['column_name']: {'type': col['data_type']} for col in schema}
        }
    }
    for model_name in gen_class.model_families.get(intent.get("task"), []):
        dataset.append({
            'dataset_name': dataset_name,
            'table_name': table_name,
            'schema': schema,
            'intent': intent,
            'model_name': model_name
    })
    checked_intents.add(k)

logger.info(f"Total dataset entries: {len(dataset)}")

table_specific_exclusions = {
    "orders": ["user_id", "order_id", "created_at", "returned_at", "shipped_at", "delivered_at"],
    "disclosures_13": ["record_id", "family_id", "blanket_scope", "disclosure_event", "pub_cleaned", "wg_name"],
    "outpatient_charges_2014": ["provider_id"],
    "fips": ["GeoName", "GeoFIPS"],
    "mbb_historical_teams_games": ["team_id", "name", "market", "opp_id", "opp_name", "opp_code", "opp_market"],
    "historical_runups": ["id", "tsevent_id", "location_name"],
    "solar_potential_by_postal_code": ["center_point", "install_size_kw_buckets"],
}

outputs = []
generator = gen_class()
for x in tqdm(dataset, total=len(dataset)):
    train_output = generator.gen(
        dataset_name=x['dataset_name'],
        table_name=x['table_name'],
        schema=x['schema'],
        intent=x['intent'],
        is_train=True,
        model_name=x['model_name'],
        exclude_cols=table_specific_exclusions.get(x['table_name'], []),
        test_size=0.1,
        auto_preprocess=True
    )
    outputs.append(train_output)

errors = []
for i, o in tqdm(enumerate(outputs), total=len(outputs)):
    # create view_table
    try:
        execute_query(o['view_table'])
        logger.info(f"Created view table with query: {o['relation_name']}")
        time.sleep(0.5)
    except Exception as e:
        logger.error(f"Error creating view table: {e}")
        errors.append({'o': o, 'error': str(e)})
    # train model
    try:
        execute_query(o['query'])
        logger.info(f"Trained model.")
        time.sleep(0.5)
    except Exception as e:
        logger.error(f"Error training model: {e}")
        errors.append({'o': o, 'error': str(e)})
    # drop view
    try:
        query = f"""
        DROP VIEW IF EXISTS {o['relation_name']};
        """
        execute_query(query)
        logger.info(f"Dropped view table: {o['relation_name']}")
        time.sleep(0.5)
    except Exception as e:
        logger.error(f"Error dropping view table: {e}")
        errors.append({'o': o, 'error': str(e)})

if errors:
    with open('errors.json', 'w') as f:
        json.dump(errors, f, indent=2)


# Get all projects results to construct a db for ML selection
query = f"""SELECT * FROM pgml.projects"""
x = execute_query(query, fetch=True)
df_projects = pd.DataFrame(x)
query = f"""SELECT * FROM pgml.models"""
y = execute_query(query, fetch=True)
df_models = pd.DataFrame(y)

df = pd.merge(df_models, df_projects, left_on='project_id', right_on='id', suffixes=('_model', '_project')).loc[:, ['name', 'snapshot_id', 'num_features', 'metrics']]
df['name'] = df['name'] + '/' + df['snapshot_id'].astype(str)
df.drop(columns=['snapshot_id'], inplace=True)
df = pd.concat([df, df['name'].str.split('/', expand=True).rename(columns={0: 'dataset', 1: 'table', 2: 'task', 3: 'algorithm', 4: 'model_id'})], axis=1)
metrics = {'r2': [], 'rmse': [], 'mae': [], 'fit_time': [], 'score_time': []}
for r in df['metrics']:
    metrics['r2'].append(r['r2'] if r['r2'] else None)
    metrics['rmse'].append(np.sqrt(r['mean_squared_error']) if r['mean_squared_error'] and (r['mean_squared_error'] < 1e10) else None)
    metrics['mae'].append(r['mean_absolute_error'] if r['mean_absolute_error'] and (r['mean_absolute_error'] < 1e10) else None)
    metrics['fit_time'].append(r['fit_time'] if r['fit_time'] else None)
    metrics['score_time'].append(r['score_time'] if r['score_time'] else None)
df = pd.concat([df, pd.DataFrame(metrics)], axis=1)
df.drop(columns=['name', 'metrics'], inplace=True)

size = {'table': [], 'num_rows': []}
for p in CSV_DIR.glob('*.csv'):
    df_temp = pd.read_csv(p)
    size['table'].append(p.stem.removesuffix("_cleaned"))
    size['num_rows'].append(df_temp.shape[0])

df = pd.merge(df, pd.DataFrame(size), on='table', how='left')
df = df.loc[:, ['dataset', 'table', 'task', 'algorithm', 'model_id', 'num_rows', 'num_features', 'r2', 'rmse', 'mae', 'fit_time', 'score_time']]
df.to_csv('./exps/exp_mlselection.csv', index=False)