import os
import re
import json
import requests
import argparse
from src import prompts
from src.extract_intent_bigqueryml import extract_intent_bigquery
from src.extract_intent_postgresml import extract_intent_postgres

from bs4 import BeautifulSoup
from loguru import logger
from langchain_openai import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.utilities import GoogleSearchAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, Tool, AgentType

from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())

class SimpleAgent:
    """gpt-5-mini Agent (stop/temperature)."""

    def __init__(self, llm: ChatOpenAI, tools: dict, max_steps: int = 3):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps

    def run(self, task: str) -> str:
        scratch = []
        for step in range(self.max_steps):
            history = "\n".join(
                f"Action: {a}\nObservation: {o}" for a, o in scratch
            )
            prompt = prompts.TYPICAL_GPT5_MINI_REACT.format(
                task=task,
                history=history if history else "(none)"
            )
            out = self.llm.invoke(prompt).content.strip()
            try:
                data = json.loads(out)
            except Exception:
                return f"[ERROR] invalid output: {out}"

            if "final_answer" in data:
                return data["final_answer"]

            act = data.get("action")
            inp = data.get("action_input")
            if act not in self.tools:
                return f"[ERROR] Unknown tool {act}"
            try:
                obs = self.tools[act](inp)
            except Exception as e:
                obs = f"[TOOL ERROR] {e}"

            scratch.append((json.dumps(data, ensure_ascii=False), str(obs)[:1000]))

        return "[ERROR] too many steps, no final answer."

class PipelineWithGPT5:
    def __init__(self, model_name: str = "gpt-5-mini", platform: str = "bigquery"):
        self.model_name = model_name
        self.platform = platform
        self.llm = ChatOpenAI(model=model_name, temperature=1)
        search = GoogleSearchAPIWrapper()

        def google_search(query: str):
            return search.results(query, 5)

        def scrape_website_full(url: str) -> str:
            clean_url = url.strip().split()[0].strip('"').strip("'")
            headers = {"User-Agent": "Mozilla/5.0"}
            res = requests.get(clean_url, headers=headers, timeout=10)
            res.raise_for_status()
            soup = BeautifulSoup(res.text, "html.parser")
            texts = soup.find_all(["p", "li", "h1", "h2", "h3"])
            return "\n".join(t.get_text(strip=True) for t in texts)

        self.tools = {
            "google_search": google_search,
            "scrape_website_full": scrape_website_full,
        }

        # ---- Simple Agent ----
        self.agent = SimpleAgent(self.llm, self.tools, max_steps=3)

    def choose_algorithm(self, query: str) -> str:
        task = (
            f"Find which BigQuery ML algorithm fits this query: {query}. "
            "Use search + scrape if needed. Only output the algorithm name in final_answer."
        )
        return self.agent.run(task)

    def generate_sql(self, query: str, algorithm: str, data_dict: str) -> str:
        doc_task = f"Summarize official syntax & hyperparams for BigQuery ML {algorithm}."
        doc_summary = self.agent.run(doc_task)

        sql_prompt = prompts.TYPICAL_BIGQUERY.format(
            query=query, algorithm=algorithm, data_dict=data_dict, doc_summary=doc_summary
        )
        raw = self.llm.invoke(sql_prompt).content.strip()
        try:
            obj = json.loads(raw)
            return json.dumps(obj, ensure_ascii=False)
        except Exception:
            return json.dumps({"training_sql": "", "inference_sql": ""})

    def run(self, query: str, data_dict: str) -> dict:
        with get_openai_callback() as cb:
            try:
                alg = self.choose_algorithm(query)
            except Exception as e:
                print(f"[ERROR] Algorithm selection failed: {e}")
                alg = "Logistic Regression"

            try:
                sql = self.generate_sql(query, alg, data_dict)
            except Exception as e:
                print(f"[ERROR] SQL generation failed: {e}")
                sql = json.dumps({"training_sql": "", "inference_sql": ""})

        return {
            "algorithm_decision": alg,
            "generated_sql": sql,
            "total_tokens": cb.total_tokens,
            "prompt_tokens": cb.prompt_tokens,
            "completion_tokens": cb.completion_tokens,
            "total_cost_usd": round(cb.total_cost, 6),
        }

class Pipeline:
    llm_classes = {
        "gpt-4o": ChatOpenAI,
        "gpt-4o-mini": ChatOpenAI,
        "gemini-2.0-flash": ChatGoogleGenerativeAI,
    }
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        platform: str = "bigquery",
        temperature: float = 0.0,
    ):
        self.model_name = model_name
        self.platform = platform
        self.llm = self.llm_classes[self.model_name](
            model_name=model_name,
            temperature=temperature,
        )
        
        search_tool = GoogleSearchAPIWrapper()
        def top5_results(query):
            return search_tool.results(query, 5)
        google_search_tool = Tool(
            name="google_search",
            func=top5_results,
            description="It is a search tool for finding related websites. Please enter a keyword."
        )

        def scrape_all_text(url: str) -> str:
            clean_url = url.split()[0].strip().strip('"').strip("'")
            headers = {'User-Agent': 'Mozilla/5.0'}
            try:
                res = requests.get(clean_url, headers=headers, timeout=10)
                res.raise_for_status()
            except Exception as e:
                return f"[ERROR] {str(e)}"

            soup = BeautifulSoup(res.text, 'html.parser')
            texts = soup.find_all(['p', 'li', 'h1', 'h2', 'h3'])
            content = '\n'.join(t.get_text(strip=True) for t in texts)
            return content

        scrape_tool = Tool(
            name="scrape_website_full",
            func=scrape_all_text,
            description="It scrapes the full text of a website. Please enter a URL."
        )

        def extract_relevant_summary(html_text: str, query: str) -> str:
            if not query or not html_text:
                return "[ERROR] 'query' and 'html_text' are required."

            llm = self.llm_classes[self.model_name](model_name=self.model_name, temperature=0.0)

            prompt = prompts.TYPICAL_WEBSEARCH_SUMMARY.format(
                query=query,
                html_text=html_text[:150000]
            )
            
            response = llm.predict(prompt)
            return response.strip()

        summary_tool = Tool(
            name="relevant_summary_tool",
            func=extract_relevant_summary,
            description="Receives the entire website text (html_text) and the user query (query), and summarizes only the relevant parts.\nThe input must be in JSON format, including both query and html_text."
        )

        tools = [google_search_tool, scrape_tool]
        self.alg_agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
        )
        self.sql_agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            handle_parsing_errors=True,
        )

    def choose_algorithm(self, query: str) -> str:
        if self.platform == "bigquery":
            return self.alg_agent.run(
                prompts.TYPICAL_BIGQUERY_ALGORITHM_SELECTION.format(query=query))
        elif self.platform == "postgres":
            return self.alg_agent.run(
                prompts.TYPICAL_POSTGRES_ALGORITHM_SELECTION.format(query=query))

    def generate_sql(self, query: str, algorithm: str, data_dict: str) -> str:
        documentation_summary = self.sql_agent.run(
            prompts.TYPICAL_WEBSEARCH.format(algorithm=algorithm))
        if self.platform == 'bigquery':
            return self.llm.predict(prompts.TYPICAL_BIGQUERY.format(
                query=query,
                algorithm=algorithm,
                data_dict=data_dict,
                documentation_summary=documentation_summary)).strip()
        elif self.platform == 'postgres':
            return self.llm.predict(prompts.TYPICAL_POSTGRES.format(
                query=query,
                algorithm=algorithm,
                data_dict=data_dict,
                documentation_summary=documentation_summary)).strip()

    def run(self, query: str, data_dict: str) -> dict:
        """
        1) Choosen ML Algorithm
        2) SQL Generation
        3) Cost and Token Calculation
        """
        with get_openai_callback() as cb:
            try:
              alg = self.choose_algorithm(query)
            except Exception as e:
              print(f"[ERROR] Algorithm selection failed: {e}")
              alg = "Logistic Regression"

            try:
              sql = self.generate_sql(query, alg, data_dict)
            except Exception as e:
              print(f"[ERROR] SQL generation failed: {e}")
              sql = json.dumps({"training_sql": "", "inference_sql": ""})

        return {
            "algorithm_decision": alg.strip(),
            "generated_sql": sql.strip(),
            "total_tokens": cb.total_tokens,
            "prompt_tokens": cb.prompt_tokens,
            "completion_tokens": cb.completion_tokens,
            "total_cost_usd": round(cb.total_cost, 6),
        }

def run(args, pipeline):
    processed = set()
    if os.path.exists(args.output_path):
        with open(args.output_path, "r", encoding="utf-8") as fout:
            for line in fout:
                try:
                    rec = json.loads(line.strip())
                    key = rec.get("question", "")
                    if key:
                        processed.add(key)
                except:
                    continue
        logger.info(f"Restarting... Already processed: {len(processed)} records")

    buffer = []
    save_every = 10

    with open(args.input_path, "r", encoding="utf-8") as fin, \
         open(args.output_path, "a", encoding="utf-8") as fout:

        for idx, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            logger.info(f"▶ Processing example {idx}...")
            record = json.loads(line)
            query = record.get("question", "")
            data_dict = record.get("schema", {})
            ground_truth = record.get("intent", {})

            if query in processed:
                continue

            result = pipeline.run(query, json.dumps(data_dict))
            raw = result.get("generated_sql", "")
            clean = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
            clean = re.sub(r"\s*```$", "", clean, flags=re.MULTILINE).strip()
            try:
                sql_obj = json.loads(result["generated_sql"])
            except json.JSONDecodeError:
                sql_obj = {"training_sql": "", "inference_sql": ""}

            if args.platform == "bigquery":
                intent = extract_intent_bigquery(
                    sql_obj["training_sql"], sql_obj["inference_sql"], "errors.txt")
            elif args.platform == "postgres":
                intent = extract_intent_postgres(
                    sql_obj["training_sql"], sql_obj["inference_sql"], "errors.txt")

            out_rec = {
                "instruction": query,
                "input": data_dict,
                "ground_truth": ground_truth,
                "algorithm_decision": result.get("algorithm_decision", ""),
                "training_sql": sql_obj.get("training_sql", ""),
                "inference_sql": sql_obj.get("inference_sql", ""),
                "total_tokens": result.get("total_tokens", 0),
                "prompt_tokens": result.get("prompt_tokens", 0),
                "completion_tokens": result.get("completion_tokens", 0),
                "total_cost_usd": result.get("total_cost_usd", 0.0),
                "intent": intent
            }

            buffer.append(out_rec)

            if len(buffer) >= save_every:
                for rec in buffer:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                os.fsync(fout.fileno())  # 디스크에 강제 반영
                logger.info(f"💾 Saved {len(buffer)} records (up to example {idx})")
                buffer = []

        if buffer:
            for rec in buffer:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            os.fsync(fout.fileno())
            logger.info(f"Final save {len(buffer)} records")

    logger.info(f"Done. Results written to {args.output_path}")

def main(args):
    if args.model_name == 'gpt-5-mini':
        pipeline = PipelineWithGPT5(model_name=args.model_name, platform=args.platform)
    elif args.model_name in ('gpt-4o-mini', 'gpt-4o', 'gemini-2.0-flash'):
        pipeline = Pipeline(model_name=args.model_name, platform=args.platform)
    else:
        raise ValueError(f"Unknown model name: {args.model_name}")
    run(args, pipeline)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument('--model_name', type=str, default='gpt-5-mini, gpt-4o-mini ...')
    parser.add_argument('--platform', type=str, default='bigquery/postgres')
    args = parser.parse_args()
    main(args)