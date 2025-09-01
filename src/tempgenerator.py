import os
import hashlib
import json
from typing import Literal, Optional, Any

class TemplateGenerator():
    comparison_ops = {">", "<", ">=", "<="}

    def __init__(self, platform_type: Literal["bigquery", "postgresql"]):
        self.platform_type = platform_type

    def extract_tag_value(self, tagged_str: str, tag: str) -> str:
        # Extracts <tag>value</tag> from a string
        if not tagged_str:
            return ""
        import re
        pattern = f"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, tagged_str)
        return match.group(1) if match else ""
    
    def format_val(self, val: str) -> str:
        try:
            float(val)
            return val
        except ValueError:
            return f"'{val}'"

    def parse_cond(self, cond: str) -> tuple[str, str, str]:
        col = self.extract_tag_value(cond, "col")
        op = self.extract_tag_value(cond, "op")
        val = self.extract_tag_value(cond, "val")
        return col, op, val
    
    def split_update_conditions(self, update_conds_raw: list[str]) -> tuple[list[str], list[str]]:
        filter_like, true_updates = [], []
        for cond in update_conds_raw or []:
            col, op, val = self.parse_cond(cond)
            if op in self.comparison_ops:
                filter_like.append(cond)
            else:
                true_updates.append(cond)
        return filter_like, true_updates
    
    def quote_ident(self, name: str) -> str:
        quote_char = '"' if self.platform_type == "postgresql" else "`"
        if name is None:
            return ''
        s = str(name).strip()
        if s.startswith(quote_char) and s.endswith(quote_char):
            return s
        s = s.replace(quote_char, f"{quote_char}{quote_char}")
        return f"{quote_char}{s}{quote_char}"
    
    def get_input_feature_columns_from_schema(self, schema: dict[str, dict[str, dict[str, Any]]], 
                                              exclude_cols: Optional[list[str]|None]) -> list[str]:
        """
        {
            <table_name>: {
                'columns': {
                    <column_name1>: dict,
                    <column_name2>: dict,
                    ...
                }
            }
        }
        """
        cols = []
        for _, columns in schema.items():
            for col in columns['columns'].keys():
                if exclude_cols is None or col not in exclude_cols:
                    cols.append(col)
        return cols
    
    def gen(self, dataset_name: str, table_name: str, schema: dict, intent: dict, is_train: bool, model_name: str, **kwargs) -> dict:
        """
        intent: the intend dictionary with five keys - `time_series`, `target_column`, `inference_condition`, `update_condition`, `task`
        model_name: the name of the model to be used
        schema: data_dictionary with {table_name: {'columns': {column_name: column_infos[dict], ... }}}
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def generate_hash(self, name: str) -> str:
        return hashlib.md5(name.encode()).hexdigest()

class BigQueryTemplateGenerator(TemplateGenerator):
    model_families = {
        "Time Series": {
            "regression": ["ARIMA_PLUS", "ARIMA_PLUS_XREG", "LINEAR_REG", "BOOSTED_TREE_REGRESSOR", "DNN_REGRESSOR", "DNN_LINEAR_COMBINED_REGRESSOR" "RANDOM_FOREST_REGRESSOR"],
            "classification": ["LOGISTIC_REG", "BOOSTED_TREE_CLASSIFIER", "DNN_CLASSIFIER", "DNN_LINEAR_COMBINED_CLASSIFIER", "RANDOM_FOREST_CLASSIFIER"],
            "clustering": ["KMEANS"],
            "anomaly_detection": ["KMEANS", "ARIMA_PLUS", "ARIMA_PLUS_XREG"]

        },
        "Non Time Series":{
            "regression": ["LINEAR_REG", "BOOSTED_TREE_REGRESSOR", "DNN_REGRESSOR", "DNN_LINEAR_COMBINED_REGRESSOR", "RANDOM_FOREST_REGRESSOR"],
            "classification": ["LOGISTIC_REG", "BOOSTED_TREE_CLASSIFIER", "DNN_CLASSIFIER", "DNN_LINEAR_COMBINED_CLASSIFIER", "RANDOM_FOREST_CLASSIFIER"],
            "clustering": ["KMEANS"],
            "anomaly_detection": ["KMEANS"]
        }
    }
    def __init__(self, platform_type, google_project_id, google_data_id):
        super().__init__(platform_type)
        self.google_project_id = google_project_id
        self.google_data_id = google_data_id

    def generate_hash(self, name: str) -> str:
        return hashlib.md5(name.encode()).hexdigest()[:6]

    def gen(self, dataset_name: str, table_name: str, schema: dict, intent: dict, is_train: bool, model_name: str, table_specific_exclusions: dict[str, list[str]]) -> str:
        # dataset_name = intent["dataset_name"]
        # table_name = intent["table_id"]
        full_table = f"`{self.google_data_id}.{dataset_name}.{table_name}`"

        output = intent["output"]
        task = output["task"]
        is_time_series = output.get("time_series", "False") == "True"
        target_col = self.extract_tag_value(output.get("target_column", ""), "col")
        inference_conds_raw = output.get("inference_condition", [])
        update_conds_raw = output.get("update_condition", [])

        model_prefix = f"{self.google_project_id}.sql_knowledge_base.{target_col or task}"
        base_model_name = f"{model_prefix}_{model_name.lower().replace('-', '_')}"
        full_model_name = f"{base_model_name}_{self.generate_hash(base_model_name)}"

        # Separate update conditions
        filter_like_updates, true_updates = [], []
        for cond in update_conds_raw:
            (filter_like_updates if self.extract_tag_value(cond, "op") in self.comparison_ops else true_updates).append(cond)

        updated_cols = {self.extract_tag_value(cond, "col") for cond in true_updates}
        filtered_inference_conds = [cond for cond in inference_conds_raw if self.extract_tag_value(cond, "col") not in updated_cols]
        all_filters = filtered_inference_conds + filter_like_updates

        where_clause = " AND ".join(
            f"{self.quote_ident(self.extract_tag_value(c, 'col'))} {self.extract_tag_value(c, 'op')} {self.format_val(self.extract_tag_value(c, 'val'))}"
            for c in all_filters
        )

        needs_label = target_col and task not in ["clustering", "anomaly_detection"]
        label_opt = f", INPUT_LABEL_COLS=['{target_col}']" if needs_label else ""
        where_filter = f"WHERE {target_col} IS NOT NULL" if needs_label else ""

        # Identify time column
        data_dict = intent.get("data_dictionary", {})
        timestamp_cols_by_priority = {"TIMESTAMP": [], "DATETIME": [], "DATE": []}
        for col, meta in data_dict.items():
            col_type = meta.get("type")
            if col_type in timestamp_cols_by_priority:
                timestamp_cols_by_priority[col_type].append(col)

        time_col = None
        if is_time_series:
            for t in ["TIMESTAMP", "DATETIME", "DATE"]:
                if timestamp_cols_by_priority[t]:
                    time_col = timestamp_cols_by_priority[t][0]
                    break
            if not time_col:
                raise ValueError("No time column found for time-series model.")
            if not target_col:
                raise ValueError("No target_col provided for time-series model.")

        # Excluded columns
        timestamp_cols = {col for col, meta in data_dict.items() if meta.get("type") in {"TIMESTAMP", "DATE", "DATETIME"}}
        excluded_cols = timestamp_cols
        if is_time_series:
            excluded_cols.update({time_col, target_col})

        excluded_cols.update(table_specific_exclusions.get(table_name, []))
        excluded_cols = {c for c in excluded_cols if c}

        exclude_clause = ", ".join(sorted(excluded_cols))
        select_clause = f"* EXCEPT({exclude_clause})" if excluded_cols else "*"

        # Subquery for inference
        subquery = f"(SELECT * FROM {full_table}"
        if where_clause:
            subquery += f" WHERE {where_clause}"
        subquery += ")"

        # Training SQL
        if is_train:
            if is_time_series:
                ts_opts = [
                    f"TIME_SERIES_TIMESTAMP_COL='{time_col}'",
                    f"TIME_SERIES_DATA_COL='{target_col}'"
                ]
                if "xreg" in model_name.lower():
                    training_sql = f"""
                    CREATE MODEL IF NOT EXISTS `{full_model_name}`
                    OPTIONS(model_type='ARIMA_PLUS_XREG', {", ".join(ts_opts)})
                    AS
                    (SELECT {time_col}, {target_col}, {select_clause} FROM {full_table}
                    {where_filter})
                    """.strip()
                else:
                    training_sql = f"""
                    CREATE MODEL IF NOT EXISTS `{full_model_name}`
                    OPTIONS(model_type='ARIMA_PLUS', {", ".join(ts_opts)})
                    AS
                    (SELECT {time_col}, {target_col} FROM {full_table}
                    {where_filter})
                    """.strip()
            else:
                training_sql = f"""
                CREATE MODEL IF NOT EXISTS `{full_model_name}`
                OPTIONS(model_type='{model_name}'{label_opt},
                DATA_SPLIT_METHOD='RANDOM',
                DATA_SPLIT_EVAL_FRACTION=0.10)
                AS
                (SELECT {select_clause} FROM {full_table}
                {where_filter})
                """.strip()
            return training_sql
        # Inference SQL
        else:
            if task == "anomaly_detection":
                inference_sql = f"""
                SELECT * FROM ML.DETECT_ANOMALIES(
                  MODEL `{full_model_name}`,
                  STRUCT(0.2 AS contamination),
                  {subquery}
                )
                """.strip()
            elif is_time_series:
                if "xreg" in model_name.lower():
                    inference_sql = f"""
                    SELECT * FROM ML.FORECAST(
                      MODEL `{full_model_name}`,
                      STRUCT(10 AS horizon, 0.8 AS confidence_level),
                      {subquery}
                    )
                    """.strip()
                elif "arima_plus" in model_name.lower():
                    inference_sql = f"""
                    SELECT * FROM ML.FORECAST(
                      MODEL `{full_model_name}`,
                      STRUCT(10 AS horizon, 0.8 AS confidence_level))
                    """.strip()
                else:
                    inference_sql = f"""
                    SELECT * FROM ML.PREDICT(MODEL `{full_model_name}`, {subquery})
                    """.strip()
            else:
                inference_sql = f"""
                SELECT * FROM ML.PREDICT(MODEL `{full_model_name}`, {subquery})
                """.strip()
            return inference_sql

def _merge_user(custom_for_col: dict, base: dict) -> dict:
    """
    Deep-ish merge: only keys present in `custom_for_col` override.
    Unknown keys are ignored (keeps output clean for downstream systems).
    """
    out = base.copy()
    for k in ("impute", "scale", "encode"):
        if k in custom_for_col:
            out[k] = custom_for_col[k]
    return out

def _flatten_schema(schema: dict) -> dict:
    """
    Accepts either:
    {<table>: {"columns": {<col>: {"type": ...}}}}
    or directly:
    {"columns": {...}}
    Returns {column_name: type_string}
    """
    # If schema already at columns level
    if "columns" in schema:
        return {c: (v.get("type") or "").upper() for c, v in schema["columns"].items()}

    # Else assume top-level tables
    out = {}
    for _tbl, spec in schema.items():
        cols = spec.get("columns", {})
        out.update({c: (v.get("type") or "").upper() for c, v in cols.items()})
    return out


class PostgresTemplateGenerator(TemplateGenerator):
    pgml_schema = os.getenv("PGML_SCHEMA", "public")
    model_families = {
        "regression": ['xgboost', 'ada_boost', 'catboost', 'random_forest', 'bagging', 'svm', 'elastic_net', 'bayesian_ridge', 'linear', 'stochastic_gradient_descent'],
        "classification": ['xgboost', 'ada_boost', 'random_forest', 'bagging', 'svm', 'ridge'],
        "clustering": ['mini_batch_kmeans'],
    }
    def __init__(self):
        super().__init__(platform_type="postgresql")


    def _gen_preprocess(self, schema: dict, exclude_cols: list[str], 
                        custom: Optional[dict|None]=None) -> dict:
        """
        https://github.com/postgresml/postgresml/blob/master/pgml-cms/docs/open-source/pgml/guides/supervised-learning/data-pre-processing.md
        custom: {column: {impute|scale|encode: ...}}
        Valid options:
            impute:  error | mean | median | mode | min | max | zero
            scale:   preserve | standard | min_max | max_abs | robust
            encode:  native | target | one_hot | ordinal
        Returns:
            {column: {impute, scale, encode}}
        """

        defaults_by_type = {
            "DOUBLE PRECISION": {"impute": "median"},
            "REAL": {"impute": "median"},
            "NUMERIC": {"impute": "median"},
            "DECIMAL": {"impute": "median"},

            "BIGINT":   {"impute": "mode"},
            "INTEGER":  {"impute": "mode"},
            "SMALLINT": {"impute": "mode"},

            "TEXT": {"impute": "mode"},
            "GEOGRAPHY":{"impute": "mode"},
        }
        
        custom = custom or {}
        col_types = _flatten_schema(schema)
        result = {}

        for col, typ in col_types.items():
            if col in exclude_cols:
                continue
            base_type = defaults_by_type.get(typ, {"impute": "mode"})
            # Merge any user overrides
            if col in custom:
                base_type = _merge_user(custom[col], base_type)
            result[col] = base_type

        return result
    
    def gen(self, dataset_name: str, table_name: str, schema: dict, intent: dict, is_train: bool, model_name: str, 
            auto_preprocess: bool=False, 
            exclude_cols: Optional[list[str]]=[], 
            test_size: Optional[float|None]=0.1, **kwargs) -> dict:
        """
        Generate the SQL queries and view definitions for the given parameters.
        
        
        ```
        train_output = generator.gen(
            dataset_name=x['dataset_name'],
            table_name=x['table_name'],
            schema=x['schema'],
            intent=x['intent'],
            is_train=True,
            model_name=x['model_name'],
            exclude_cols=table_specific_exclusions.get(x['table_name'], [])
        )
        inference_output = generator.gen(
            dataset_name=x['dataset_name'],
            table_name=x['table_name'],
            schema=x['schema'],
            intent=x['intent'],
            is_train=False,
            model_name=x['model_name'],
            exclude_cols=table_specific_exclusions.get(x['table_name'], [])
        )
        ```
        
        """
        # assert the model_name is the algorithm name
        assert model_name in self.model_families.get(intent.get("task"), []), f"Model {model_name} is not suitable for task {intent.get('task')}"

        task = (intent.get("task") or "").lower()
        is_time_series = intent.get("time_series", "False") == "True"
        target_col = self.extract_tag_value(intent.get("target_column", ""), "col")

        # Generate unique model/project name
        project_name = f"{dataset_name}/{table_name}/{task}/{model_name}/{target_col or 'no_target'}"
        short_name = hashlib.md5(project_name.encode()).hexdigest()

        input_feature_cols = self.get_input_feature_columns_from_schema(schema, [target_col]+exclude_cols)

        # === TRAINING ===
        if is_train:
            # Parse input features
            relation_name = f"{self.pgml_schema}.{self.quote_ident(short_name)}"
            # Create a view table for training.
            tuple_expr = ", ".join(self.quote_ident(c) for c in input_feature_cols)
            target_expr = ', ' + self.quote_ident(target_col) if target_col else ''
            view_table_query = f"""CREATE OR REPLACE VIEW {relation_name} AS (SELECT {tuple_expr}{target_expr} FROM {self.pgml_schema}.{table_name});"""
            
            # Preprocess arguments
            if auto_preprocess:
                preprocess = self._gen_preprocess(schema, exclude_cols=[target_col]+exclude_cols, **kwargs)
            else:
                preprocess = kwargs['custom'] if kwargs.get('custom') else {}
            if task in ("regression", "classification", "clustering"):
                query = self.build_pgml_train_sql(
                    project_name=project_name,
                    task=task,
                    relation_name=relation_name.replace('"', ''),
                    y_column_name=target_col,
                    algorithm=model_name,
                    preprocess=preprocess,
                    test_size=test_size
                )
            else:
                raise ValueError(f"Unsupported task: {task}")
            
            return {
                'project_name': project_name,
                'relation_name': relation_name,
                'view_table': view_table_query.strip(),
                'query': query.strip(),
                'preprocess': preprocess,
            }

        # === INFERENCE ===
        else:
            # Inference on the original table with filters but selection should be same as training
            relation_name = f"{self.pgml_schema}.{self.quote_ident(table_name)}"
            
            # Parse conditions
            inference_conds_raw = sorted(intent.get("inference_condition", []))
            update_conds_raw = sorted(intent.get("update_condition", []))
            filter_like_updates, true_updates = self.split_update_conditions(update_conds_raw)
            where_clause = []
            for c in filter_like_updates:
                col_u, op_u, val_u = self.parse_cond(c)
                for i, ci in enumerate(inference_conds_raw):
                    col_i, op_i, val_i = self.parse_cond(ci)

                    # check if both conditions are on the same column and operation but different values(take the update one)
                    if col_i == col_u and op_i == op_u and val_i != val_u:
                        where_clause.append(f"{self.quote_ident(col_u)} {op_u} {self.format_val(val_u)}")
                        inference_conds_raw.pop(i)

            # append the rest of the inference conds
            for c in inference_conds_raw:
                col_i, op_i, val_i = self.parse_cond(c)
                where_clause.append(f"{self.quote_ident(col_i)} {op_i} {self.format_val(val_i)}")
            where_clause = " AND ".join(where_clause)

            if true_updates:
                query = self.build_pgml_predict_sql_scenario2(
                    project_name=project_name,
                    relation_name=relation_name,
                    input_feature_cols=input_feature_cols if input_feature_cols else [],
                    where_clause=where_clause,
                    true_updates=true_updates,
                    limit=10
                )
            else:
                query = self.build_pgml_predict_sql_scenario1(
                    project_name=project_name,
                    relation_name=relation_name,
                    input_feature_cols=input_feature_cols,
                    where_clause=where_clause,
                    limit=10
                )
                
            return {
                'query': query.strip(),
            }

    def build_pgml_train_sql(self, project_name: str, task: str, relation_name: str, y_column_name: str, algorithm: str, 
                             preprocess: dict | None = None, test_size: float | None = None):
        """
        preprocess: 
            impute: `error`, `mean`, `median`, `mode`, `min`, `max`, `zero`
            scale: `preserve`, `standard`, `min_max`, `max_abs`, `robust`
            encode: `native`, `target`, `one_hot`, `ordinal`
        """
        args = [
            f"project_name => '{project_name}'",
            f"task => '{task}'",
            f"relation_name => '{relation_name}'",
            f"algorithm => '{algorithm}'"
        ]
        if task != 'clustering':
            args.append(f"y_column_name => '{y_column_name}'")

        if preprocess:
            args.append(f"preprocess => '{json.dumps(preprocess, ensure_ascii=False)}'")
        if test_size is not None:
            args.append(f"test_size => {test_size}")
        if task == 'classification':
            args.append("test_sampling => 'stratified'")
        return "SELECT pgml.train(\n    " + ",\n    ".join(args) + "\n);\n"

    def build_pgml_predict_sql_scenario1(self, project_name: str, relation_name: str, input_feature_cols: list[str], 
                                         where_clause: str = "", limit: int | None = None):
        """
        Predict referencing columns directly (simple; no overrides).
        """
        tuple_expr = ", ".join(self.quote_ident(c) for c in input_feature_cols)
        sql = (
            "SELECT pgml.predict(\n"
            f"    '{project_name}', ({tuple_expr})\n"
            f") AS prediction\n"
            f"FROM {relation_name}"
        )
        if where_clause:
            sql += f"\nWHERE {where_clause}"
        if limit:
            sql += f"\nLIMIT {limit}"
        sql += ";\n"
        return sql

    def build_pgml_predict_sql_scenario2(self, project_name: str, relation_name: str, input_feature_cols: list[str], 
                                         where_clause: str = "", true_updates: list[str] | None = None, limit: int | None = None) -> str:
        select_items = []
        true_updates = true_updates or []
        override_map = {self.parse_cond(c)[0]: self.parse_cond(c)[2] for c in true_updates}
        for col in input_feature_cols:
            if col in override_map:
                select_items.append(f"{self.format_val(override_map[col])} AS {self.quote_ident(col)}")
            else:
                select_items.append(f"{self.quote_ident(col)}")

        inner_sql = "SELECT " + ", ".join(select_items) + f" FROM {relation_name}"
        if where_clause:
            inner_sql += f" WHERE {where_clause}"
        if limit:
            inner_sql += f" LIMIT {limit}"

        return (
            "SELECT pgml.predict(\n"
            f"    '{project_name}', t\n"
            ")\n"
            "FROM (\n"
            f"    {inner_sql}\n"
            ") AS t;\n"
        )