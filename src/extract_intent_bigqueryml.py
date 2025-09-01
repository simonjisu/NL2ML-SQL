# -*- coding: utf-8 -*-
import json
import re
import logging
from pathlib import Path
from sqlglot import parse_one, exp
from sqlglot.errors import ParseError


def _tag_col(x: str) -> str:
    return f"<col>{x}</col>"

def _val_sql(e: exp.Expression) -> str:
    try:
        return e.sql(dialect="bigquery")
    except Exception:
        return str(e)

_CONST_FUNCS = {
    "DATE", "DATETIME", "TIME", "TIMESTAMP", "TIMESTAMP_MILLIS",
    "TIMESTAMP_MICROS", "TIMESTAMP_SECONDS", "INTERVAL"
}
def _is_constant_like(e: exp.Expression) -> bool:
    if e is None:
        return False
    if isinstance(e, exp.Literal):
        return True
    if isinstance(e, exp.Paren):
        return _is_constant_like(e.this)
    if isinstance(e, (exp.Cast, exp.TryCast)):
        return _is_constant_like(e.this)
    if isinstance(e, exp.Array):
        return all(_is_constant_like(x) for x in e.expressions or [])
    if isinstance(e, exp.Struct):
        return all(_is_constant_like(x) for x in e.expressions or [])
    if isinstance(e, exp.Func):
        name = (e.name or "").upper()
        if name in _CONST_FUNCS:
            return all(_is_constant_like(x) for x in e.expressions or [])
        return False
    return False

# ---------- preprocess ----------
_SANITIZE_LINE_PAT = re.compile(
    r"^\s*(DECLARE|SET|LET|BEGIN|END|CALL)\b.*$",
    re.IGNORECASE | re.MULTILINE
)
def sanitize_sql(sql: str) -> str:
    if not sql:
        return ""
    s = sql
    s = _SANITIZE_LINE_PAT.sub("", s)
    s = re.sub(r"--.*?$", "", s, flags=re.MULTILINE)
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)
    s = s.replace(">>'", "").replace(">>`", "").replace(">>", "").replace("<<", "")
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"\{\{.*?\}\}", "", s)
    s = re.sub(r"\{[A-Za-z0-9_]+\}", "", s)
    parts = s.rsplit(";", 1)
    if len(parts) == 2 and parts[1].strip() and not any(
        k in parts[1].upper() for k in ("SELECT", "CREATE", "WITH", "INSERT", "UPDATE", "DELETE", "ML.", "FROM")
    ):
        s = parts[0] + ";"
    return s.strip()

# ---------- Find ML Function ----------
_ML_FUNCS = {"ML.PREDICT", "ML.FORECAST", "ML.DETECT_ANOMALIES"}
def _find_ml_input_select_ast(inference_sql: str):
    try:
        tree = parse_one(inference_sql, read="bigquery")
    except ParseError:
        return None
    for f in tree.find_all(exp.Func):
        name = (f.name or "").upper()
        if name in _ML_FUNCS:
            # 두 번째 인자에서 SELECT를 찾는다
            for arg in f.expressions:
                if isinstance(arg, exp.Subquery) and isinstance(arg.this, exp.Select):
                    return arg.this
                if isinstance(arg, exp.Paren) and isinstance(arg.this, exp.Select):
                    return arg.this
                if isinstance(arg, exp.Select):
                    return arg
    return None

# ---------- Normalize WHERE/QUALIFY/HAVING ----------
OP_MAP = {
    "EQ": "=", "NEQ": "!=", "GT": ">", "LT": "<", "GTE": ">=", "LTE": "<=",
    "Like": "LIKE", "ILike": "ILIKE", "In": "IN"
}

def _flatten_boolean(expr, out):
    if expr is None:
        return
    if isinstance(expr, exp.Paren):
        _flatten_boolean(expr.this, out)
        return
    if isinstance(expr, exp.And):
        _flatten_boolean(expr.left, out)
        _flatten_boolean(expr.right, out)
        return
    if isinstance(expr, exp.Between):
        col = _val_sql(expr.this)
        lo  = _val_sql(expr.args["low"])
        hi  = _val_sql(expr.args["high"])
        out.append((_tag_col(col), ">=", lo))
        out.append((_tag_col(col), "<=", hi))
        return
    if isinstance(expr, exp.In):
        col = _val_sql(expr.this)
        # 리스트를 그대로 SQL 문자열로 유지
        vals = _val_sql(expr.expression)
        out.append((_tag_col(col), "IN", vals))
        return
    if isinstance(expr, (exp.Like, exp.ILike)):
        col = _val_sql(expr.this)
        pat = _val_sql(expr.expression)
        op = "ILIKE" if isinstance(expr, exp.ILike) else "LIKE"
        out.append((_tag_col(col), op, pat))
        return
    if isinstance(expr, exp.Is):
        col = _val_sql(expr.this)
        rhs = _val_sql(expr.expression)
        raw = _val_sql(expr)
        op = "IS NOT" if " IS NOT " in raw.upper() else "IS"
        out.append((_tag_col(col), op, rhs))
        return
    opn = expr.__class__.__name__
    if opn in OP_MAP:
        left = _val_sql(expr.left)
        right = _val_sql(expr.right)
        out.append((_tag_col(left), OP_MAP[opn], right))
        return

def _collect_conditions_from_select(select_ast: exp.Select):
    conds = []
    # top-level WHERE
    where = select_ast.args.get("where")
    if isinstance(where, exp.Where):
        _flatten_boolean(where.this, conds)
    # qualify = select_ast.args.get("qualify")
    # if isinstance(qualify, exp.Qualify):
    #     _flatten_boolean(qualify.this, conds)
    # having = select_ast.args.get("having")
    # if isinstance(having, exp.Having):
    #     _flatten_boolean(having.this, conds)
    return conds

# ===================== extract_intent =====================
TIME_SERIES_MODELS = {"ARIMA", "ARIMA_PLUS", "ARIMA_PLUS_XREG"}
REGRESSION_MODELS = TIME_SERIES_MODELS | {
    "LINEAR_REG", "BOOSTED_TREE_REGRESSOR", "DNN_REGRESSOR",
    "DNN_LINEAR_COMBINED_REGRESSOR", "AUTOML_REGRESSOR",
    "RANDOM_FOREST_REGRESSOR"
}
CLASSIFICATION_MODELS = {
    "LOGISTIC_REG", "BOOSTED_TREE_CLASSIFIER", "DNN_CLASSIFIER",
    "DNN_LINEAR_COMBINED_CLASSIFIER", "AUTOML_CLASSIFIER",
    "RANDOM_FOREST_CLASSIFIER"
}

def extract_intent(training_sql: str, inference_sql: str) -> dict:
    intent = {
        "time_series": "",
        "target_column": "",
        "inference_condition": [],
        "update_condition": [],
        "task": ""
    }

    up = (inference_sql or "").upper()
    if "ML.DETECT_ANOMALIES" in up:
        ml_func = "detect_anomalies"
    elif "ML.FORECAST" in up:
        ml_func = "forecast"
    elif "ML.PREDICT" in up:
        ml_func = "predict"
    else:
        ml_func = None

    model_type = None
    if training_sql:
        m = re.search(r"model_type\s*=\s*'([^']+)'", training_sql, re.IGNORECASE)
        if m:
            model_type = m.group(1).upper()

    if ml_func == "detect_anomalies":
        intent["task"] = "anomaly_detection"
    elif ml_func in {"predict", "forecast"} and model_type:
        if model_type == "KMEANS":
            intent["task"] = "clustering"
        elif model_type in CLASSIFICATION_MODELS:
            intent["task"] = "classification"
        elif model_type in REGRESSION_MODELS:
            intent["task"] = "regression"

    if (ml_func == "forecast") or (model_type in TIME_SERIES_MODELS if model_type else False):
        intent["time_series"] = "True"
        if not intent["task"]:
            intent["task"] = "regression"
    else:
        intent["time_series"] = "False"

    opt_key = None
    if ml_func == "detect_anomalies":
        opt_key = "time_series_data_col"
    elif ml_func in {"predict", "forecast"}:
        opt_key = "time_series_data_col" if model_type in TIME_SERIES_MODELS else "input_label_cols"
    if opt_key:
        arr = re.search(rf"{opt_key}\s*=\s*\[\s*'([^']+)'.*?\]\s*", training_sql or "", re.IGNORECASE)
        if arr:
            labels = re.findall(r"'([^']+)'", arr.group(0))
            intent["target_column"] = [_tag_col(x) for x in labels] if len(labels) > 1 else _tag_col(labels[0])
        else:
            single = re.search(rf"{opt_key}\s*=\s*'([^']+)'", training_sql or "", re.IGNORECASE)
            if single:
                intent["target_column"] = _tag_col(single.group(1))

    # ----- ML 입력 SELECT에서 inference_condition / update_condition 추출 -----
    select_ast = _find_ml_input_select_ast(inference_sql or "")

    # (A) inference_condition: top-level WHERE
    if select_ast is not None:
        try:
            conds = _collect_conditions_from_select(select_ast)
            for col_tag, op, rhs in conds:
                intent["inference_condition"].append(f"{col_tag}<op>{op}</op><val>{rhs}</val>")
        except Exception:
            pass
    else:
        # 정규식 폴백
        sub = re.search(
            r"ML\.\w+\s*\(\s*MODEL\s+`[^`]+`\s*,\s*\(\s*(SELECT[\s\S]+?)\)\s*\)",
            inference_sql or "", flags=re.IGNORECASE
        )
        if sub:
            try:
                inf_sel = parse_one(sub.group(1), read="bigquery")
                if isinstance(inf_sel, exp.Select):
                    conds = _collect_conditions_from_select(inf_sel)
                    for col_tag, op, rhs in conds:
                        intent["inference_condition"].append(f"{col_tag}<op>{op}</op><val>{rhs}</val>")
            except ParseError:
                pass

    # (B) update_condition: 상수 AS 별칭만 추출
    if select_ast is not None:
        try:
            for item in select_ast.expressions or []:
                if isinstance(item, exp.Alias):
                    alias = item.alias
                    val_expr = item.this
                    if _is_constant_like(val_expr):
                        intent["update_condition"].append(f"{_tag_col(alias)}<op>=</op><val>{_val_sql(val_expr)}</val>")
        except Exception:
            pass
    else:
        # 오래된 가정: 두 번째 SELECT
        try:
            tree = parse_one(inference_sql or "", read="bigquery")
            selects = list(tree.find_all(exp.Select))
            if len(selects) >= 2:
                inner = selects[1]
                for item in inner.expressions or []:
                    if isinstance(item, exp.Alias) and _is_constant_like(item.this):
                        alias = item.alias
                        intent["update_condition"].append(f"{_tag_col(alias)}<op>=</op><val>{_val_sql(item.this)}</val>")
        except ParseError:
            pass

    return intent

# =====================[ callback ]=====================
def heuristic_intent(train_sql: str, infer_sql: str):
    T = (train_sql or "").upper()
    I = (infer_sql or "").upper()

    def detect_task(txt: str) -> str | None:
        if "KMEANS" in txt or "CLUSTER" in txt or "CLUSTERING" in txt:
            return "clustering"
        if "FORECAST" in txt or "TIME_SERIES" in txt:
            return "time_series_regression"
        if "CLASSIFIER" in txt or "LOGISTIC_REG" in txt:
            return "classification"
        if "REGRESSOR" in txt or "LINEAR_REG" in txt or "RANDOM_FOREST_REGRESSOR" in txt or "AUTOML_REGRESSOR" in txt:
            return "regression"
        return None

    model_type = None
    m = re.search(r"MODEL_TYPE\s*=\s*'?\s*([A-Z_]+)\s*'?", T, re.IGNORECASE)
    if m:
        model_type = m.group(1).upper()

    label = None
    m = re.search(r"INPUT_LABEL_COLS\s*=\s*\[\s*'?\s*([A-Za-z0-9_\.]+)\s*'?\s*\]", T, re.IGNORECASE)
    if m:
        label = m.group(1)

    task = detect_task(T) or detect_task(I)
    is_ts = (task == "time_series_regression") or (model_type in TIME_SERIES_MODELS if model_type else False) or ("FORECAST" in I)

    return {
        "time_series": "True" if is_ts else "False",
        "target_column": _tag_col(label) if label else "",
        "inference_condition": [],
        "update_condition": [],
        "task": ("regression" if task == "time_series_regression" else (task or "")),
    }

def extract_intent_bigquery(training_sql: str, inference_sql: str, error_log_path: str):
    t_sql = sanitize_sql(training_sql or "")
    i_sql = sanitize_sql(inference_sql or "")
    try:
        return extract_intent(t_sql, i_sql)
    except Exception as e:
        with open(error_log_path, "a", encoding="utf-8") as ef:
            ef.write("[extract_intent ERROR] " + repr(e) + "\n")
            ef.write("---- training_sql (sanitized) ----\n" + t_sql[:2000] + "\n")
            ef.write("---- inference_sql (sanitized) ----\n" + i_sql[:2000] + "\n\n")
        return heuristic_intent(t_sql, i_sql)

if __name__ == "__main__":
    training_sql = """CREATE OR REPLACE MODEL `your_project.your_dataset.taxi_trip_outliers_model`
OPTIONS(
  model_type='ARIMA',
  time_series_timestamp_col='trip_start_timestamp',
  time_series_data_col='trip_total',
  time_series_id_col='taxi_id',
  auto_arima=True
) AS SELECT trip_start_timestamp, trip_total FROM `your_project.your_dataset.taxi_trips` WHERE dropoff_location = ST_GEOGPOINT(-87.336594, 40.986718) AND pickup_longitude < -87.539915801;"""
    inference_sql = """SELECT *
FROM ML.PREDICT(
  MODEL `your_project.your_dataset.taxi_trip_outliers_model`,
  (
    SELECT
      age,
      'Married-civ-spouse' AS marital_status,
      `capital-gain` AS capital_gain,
      `capital-loss` AS capital_loss,
      `hours-per-week` AS hours_per_week,
      trip_total
    FROM `your_project.your_dataset.taxi_trips`
    WHERE trip_start_timestamp BETWEEN '2013-06-09 14:00:00 UTC' AND '2013-06-10 14:00:00 UTC'
  )
);"""
    intent = extract_intent_bigquery(training_sql, inference_sql)
    print(intent)
