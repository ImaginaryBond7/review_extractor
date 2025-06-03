import json
import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    logging.error("OPENAI_API_KEY not set. Add it to your environment or .env file.")
    raise RuntimeError("OPENAI_API_KEY not found.")


openai_client_sync = OpenAI()
openai_client_async = AsyncOpenAI()

# ------------ file helpers ------------

def load_json(filepath: str | Path):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, filepath: str | Path):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def save_csv(rows: list[list], headers: list[str], filepath: str | Path):
    import csv
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)