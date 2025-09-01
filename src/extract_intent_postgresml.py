import re
from typing import Dict, List
from sqlglot import parse_one, exp
from sqlglot.errors import ParseError

OPERATOR_MAPPING = {
    "EQ": "=",
    "NEQ": "!=",
    "GT": ">",
    "LT": "<",
    "GTE": ">=",
    "LTE": "<="
}

REGRESSION_ALGOS = {
    "linear_regression", "ridge", "lasso", "elastic_net",
    "random_forest_regressor", "xgboost", "lightgbm_regressor", "catboost_regressor",
    "svr", "knn_regressor", "dnn_regressor"
}
CLASSIFICATION_ALGOS = {
    "logistic_regression", "random_forest_classifier", "lightgbm_classifier",
    "catboost_classifier", "xgboost_classifier", "svc", "knn_classifier", "dnn_classifier"
}

def _parse_named_args_from_train(sql: str) -> Dict[str, str]:
    """
    pgml.train(...) 안의 named-arg들을 단순 정규식으로 파싱.
    예: project_name => 'dpp', y_column_name => 'price', relation_name => 'schema.table'
    문자열 리터럴은 따옴표 제거, 나머지는 원문 유지.
    """
    m = re.search(r"pgml\.train\s*\(([\s\S]*?)\)", sql, flags=re.IGNORECASE)
    if not m:
        return {}

    args_str = m.group(1)

    pairs = re.findall(r"(\w+)\s*=>\s*('([^']*)'|[^,)\n]+)", args_str, flags=re.IGNORECASE)
    out = {}
    for key, raw_val, quoted in pairs:
        key = key.strip()
        if quoted is not None:
            val = quoted  # 따옴표 제거된 순수 값
        else:
            val = raw_val.strip()
        out[key.lower()] = val
    return out

def _flatten_and_collect_conditions(node: exp.Expression) -> List[exp.Expression]:
    """
    WHERE 절의 조건 트리를 평탄화하여 비교 식 리스트로 반환.
    AND를 재귀적으로 풀어헤치고, BETWEEN은 별도 처리 대신 상위 함수에서 감지.
    """
    if node is None:
        return []

    if isinstance(node, exp.And):
        return _flatten_and_collect_conditions(node.left) + _flatten_and_collect_conditions(node.right)

    return [node]

def _collect_inference_conditions(where_expr: exp.Expression) -> List[str]:
    """
    WHERE 절에서 비교식을 추출하여
    <col>COL</col><op>OP</op><val>VAL</val> 형식으로 리스트를 반환.
    """
    results: List[str] = []
    if not where_expr:
        return results

    if isinstance(where_expr, exp.Between):
        c = where_expr.this.sql(dialect="postgres")
        lo = where_expr.args["low"].sql(dialect="postgres")
        hi = where_expr.args["high"].sql(dialect="postgres")
        results.append(f"<col>{c}</col><op>>=</op><val>{lo}</val>")
        results.append(f"<col>{c}</col><op><=</op><val>{hi}</val>")
        return results

    for n in _flatten_and_collect_conditions(where_expr):
        if isinstance(n, exp.Between):
            c = n.this.sql(dialect="postgres")
            lo = n.args["low"].sql(dialect="postgres")
            hi = n.args["high"].sql(dialect="postgres")
            results.append(f"<col>{c}</col><op>>=</op><val>{lo}</val>")
            results.append(f"<col>{c}</col><op><=</op><val>{hi}</val>")
            continue

        opn = n.__class__.__name__
        if opn in OPERATOR_MAPPING:
            try:
                col = n.left.sql(dialect="postgres")
                val = n.right.sql(dialect="postgres")
            except Exception:
                continue
            results.append(f"<col>{col}</col><op>{OPERATOR_MAPPING[opn]}</op><val>{val}</val>")

    return results

def _collect_update_conditions(select_node: exp.Select) -> List[str]:
    """
    extract SELECT list from the subquery and extract Alias(AS ...)
    and transform to <col>alias</col><op>=</op><val>rhs</val>.
    rhs: Literal/Column/Function call
    """
    updates: List[str] = []
    if not isinstance(select_node, exp.Select):
        return updates

    for expr_item in select_node.expressions:
        if isinstance(expr_item, exp.Alias):
            alias = expr_item.alias
            rhs_sql = expr_item.this.sql(dialect="postgres")
            updates.append(f"<col>{alias}</col><op>=</op><val>{rhs_sql}</val>")
    return updates

def _infer_task_from_algorithm(alg: str) -> str:
    alg_l = (alg or "").lower()
    if alg_l in REGRESSION_ALGOS:
        return "regression"
    if alg_l in CLASSIFICATION_ALGOS:
        return "classification"
    return ""

def extract_intent_postgres(training_sql: str, inference_sql: str) -> dict:
    """
    Format:
    {
        "time_series": "",
        "target_column": "",
        "inference_condition": [ "<col>..</col><op>..</op><val>..</val>", ... ],
        "update_condition": [ "<col>..</col><op>=</op><val>..</val>", ... ],
        "task": "regression|classification|clustering|anomaly_detection|..."
    }
    """
    intent = {
        "time_series": "",
        "target_column": "",
        "inference_condition": [],
        "update_condition": [],
        "task": ""
    }

    train_args = _parse_named_args_from_train(training_sql)
    task = (train_args.get("task") or "").strip().lower()
    ycol = (train_args.get("y_column_name") or "").strip()
    algo = (train_args.get("algorithm") or "").strip()

    if task:
        intent["task"] = task
    elif algo:
        intent["task"] = _infer_task_from_algorithm(algo)

    if ycol:
        intent["target_column"] = ycol

    try:
        tree = parse_one(inference_sql, read="postgres")
        subq = tree.find(exp.Subquery)
        inner_select = subq.this if (subq and isinstance(subq.this, exp.Select)) else None

        if inner_select:
            where_node = inner_select.args.get("where")
            if isinstance(where_node, exp.Where):
                intent["inference_condition"] = _collect_inference_conditions(where_node.this)

            intent["update_condition"] = _collect_update_conditions(inner_select)

    except ParseError:
        pass

    return intent


if __name__ == "__main__":
    training_sql = """
    SELECT pgml.train(
        project_name => 'dc',
        task => 'clustering',
        relation_name => 'pgml.diamonds_train',
        algorithm => 'affinity_propagation'
    );
    """
    inference_sql = """
    SELECT pgml.predict(
        'dc',
        t
    )
    FROM (
        SELECT carat, cut, color, clarity, depth, table_, x, y, z
        FROM pgml.diamonds_test
        WHERE age = 45
    ) AS t
    """

    intent = extract_intent_postgres(training_sql, inference_sql)
    print(intent)

    # training_sql = """
    # CREATE VIEW pgml.adult_train AS
    #     SELECT
    #         age, workclass, education, educational-num, marital-status, occupation, relationship, race, gender, capital-gain, capital-loss, hours-per-week, native-country, income
    #     FROM adult WHERE "index" <= 100;
    # SELECT pgml.train(
    #     project_name => 'income_prediction',
    #     task => 'classification',
    #     relation_name => 'pgml.adult_train',
    #     y_column_name => 'income',
    #     algorithm => 'logistic_regression'
    # );"""


    # inference_sql = """
    # SELECT pgml.predict('income_prediction', t)
    # FROM (
    #   SELECT age, workclass, fnlwgt, education, educational-num, marital-status, occupation, relationship, race, gender, capital-gain, capital-loss, hours-per-week, native-country
    #   FROM adult
    #   WHERE age = 45 ) AS t;"""
