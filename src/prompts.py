GEN_IMPLICIT_QUESTION = """You are a linguistic expert in conversational question rephrasing.
Please rewrite the following question in fluent, natural language.
Keep any hypothetical or conditional phrasing such as 'if X changes from A to B' or 'if X changes from <op> than A to <op> than B'.
Remove all formatting tags like <col>, <val>, <op>, but retain the meaning.
Do NOT include any extra commentary or repeat the original. Output only the rephrased question.
"Original Question: {mapped_question}
"""

ROUTE = """You are a specialized routing assistant for data questions. Your task is to classify a natural language question into one of the following categories:

1. RETRIEVAL: Questions that can be answered through operations on existing data without creating new models or insights. This includes:
   - Data selection/filtering (finding specific records)
   - Aggregations (count, sum, average, min, max)
   - Grouping operations
   - Sorting and ranking
   - Basic calculations using existing columns
   - Time-based filtering or aggregation
   - Basic joins between data sources
   - Finding existing patterns explicitly present in the data

2. MACHINE LEARNING: Questions requiring algorithms to identify patterns, make predictions, or generate insights not explicitly stored in the data. This includes:
   - Predictions about future events or values
   - Classifications of data into categories without explicit rules
   - Clustering to identify natural groupings in data
   - Anomaly/outlier detection
   - Pattern recognition beyond simple correlations
   - Natural language processing beyond keyword matching
   - Forecasting and trend projection
   - Questions about what WILL or MIGHT happen (predictions)
   - Questions requiring understanding of complex, non-linear relationships
   - Root cause analysis that isn't explicitly documented
   - Identification of hidden patterns or insights

---

For the following question, think through your classification step by step:
1. First, analyze what the question is asking for
2. Consider what data operations would be needed to answer it
3. Evaluate whether these operations match RETRIEVAL or MACHINE LEARNING criteria
5. Provide your final answer

Question: "{nlq}"

After showing your step-by-step reasoning, end with a JSON format:
{{ 
  "rationale": <str, step by step reasoning>,
  "label": <int: whether the nlq is retrieval(0) or machine learning(1)> 
}}
"""

INTENT_INSTRUCTION = """We aim to extract structured machine learning configuration arguments and conditions from a natural language question and a given data dictionary.
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
{question}

### Data Dictionary:
{schema}
"""

INTENT_TRAIN = """
### Chain of Thought:
{cot}

### Output:
{intent}
"""

TYPICAL_GPT5_MINI_REACT = """
You are a careful BigQuery ML research assistant.

TOOLS:
- google_search: input a query string, returns search results
- scrape_website_full: input a URL, returns page text
- final_answer: when you are done

Rules:
- Respond with JSON only.
- Format: {{"action":"tool_name","action_input":"..."}}
- OR:     {{"final_answer":"..."}}

Task:
{task}

Previous steps:
{history}
"""

TYPICAL_BIGQUERY = """You are a BigQuery ML expert.
User's task:\n{query}\n
Selected algorithm:\n{algorithm}\n
Data dictionary:\n{data_dict}\n
Reference documentation:\n{doc_summary}\n
Using the reference and selected algorithm, generate:
1. BigQuery ML training SQL template
2. BigQuery ML inference SQL template

Return the result as a JSON object with two fields: "training_sql" and "inference_sql.
Do not include any markdown formatting (no triple backticks or ```json blocks)\n
Example:
{
   "training_sql": "CREATE MODEL ...;",
   "inference_sql": "SELECT ... FROM ML.PREDICT(...);"
}
"""

TYPICAL_WEBSEARCH_SUMMARY = """Below is the full text extracted from a website. Please summarize only the parts relevant to the user query.
[User Query]
{query}
[Website Content]
{html_text}  # Truncated within token limit (about 100K characters)
Summary:
"""

TYPICAL_BIGQUERY_ALGORITHM_SELECTION = """Summarize the types of machine learning models supported by BigQuery ML based on the latest offical documentation.
Retrieve the content from the official site and provide a concise summary in table form.
Based on that, select the most suitable algorithm to solve the user's query
User Query:\n{query}
You don't need explanation. Just return ML algorithm."""

TYPICAL_WEBSEARCH = """Use `google_search` to find documentation or blogs about BigQuery ML syntax, hyperparameters, and usage examples for {algorithm}.
Then for promising URLs, fetch full content with `scrape_website_full` and extract relevant info.
Finally, summarize the key SQL requirements and return it for template generation.
"""

TYPICAL_POSTGRES_ALGORITHM_SELECTION = """Belowing is the types of machine learning models supported by Postgres ML.
regression:
- ridge
- lasso
- elastic_net
- least_angle
- lasso_least_angle
- orthogonal_matching_pursuit
- bayesian_ridge
- automatic_relevance_determination
- stochastic_gradient_descent
- passive_aggressive
- ransac
- theil_sen
- huber
- svm
- nu_svm
- linear_svm
- ada_boost
- bagging
- extra_trees
- gradient_boosting_trees
- random_forest

classification:
- ridge
- svm
- nu_svm
- linear_svm
- ada_boost
- bagging
- extra_trees
- gradient_boosting_trees
- random_forest
- xgboost
- catboost

clustering:
- affinity_propagation
- birch
- kmeans
- mini_batch_kmeans

Based on this, select the most suitable algorithm to solve the user's query
User Query:\n{query}
You don't need explanation. Just return ML algorithm"""

TYPICAL_POSTGRES = """You are a Postgres ML expert.\n
User's task:\n{query}\n
Selected algorithm:\n{algorithm}\n
Data dictionary:\n{data_dict}\n
Reference documentation:\n{documentation_summary}\n
Using the reference and selected algorithm, generate:
1. Postgres ML training SQL template
2. Postgres ML inference SQL template\n
Return the result as a JSON object with two fields: "training_sql" and "inference_sql".
Generate only training and inference SQL. You dont need to create table definitions.
Do not include any markdown formatting (no triple backticks or ```json blocks)
training SQL should include creating the view table.
Example:
   CREATE VIEW pgml.diamonds_train AS
      SELECT
         carat, cut, color, clarity, "depth", "table_", x, y, z, price
      FROM diamonds WHERE "index" <= 100;
   SELECT pgml.train(
      project_name => 'dpp',
      task => 'regression',
      relation_name => 'pgml.diamonds_train',
      y_column_name => 'price',
      algorithm => 'xgboost',
   );

   SELECT pgml.predict('dpp', t)
   -- data selection part
   FROM (
      SELECT 0.5 AS "carat", cut, color, clarity, "depth", "table_", x, y, z FROM pgml.diamonds_test
      WHERE cut = 'Ideal'
   ) AS t;
"""