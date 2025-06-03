from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List

from .utils import openai_client_async
from spacy.lang.en import English

# ---------------- spaCy fallback ----------------
_nlp = English()
_nlp.add_pipe("sentencizer")
_FALLBACK_KEYWORDS: set[str] = {
    "brand", "fragrance", "fresh", "quality", "price", "durable", "comfortable",
}

# ---------------- OpenAI helper ----------------

_SYSTEM_PROMPT = """ You are a product-feedback analysis assistant.

TASK  
Given ONE customer-review text, return a **single JSON object** exactly in this form:

{
  "attributes": ["<noun phrase>", …]
}

CONSTRAINTS
• List only the POSITIVE product attributes the reviewer clearly praises.  
• Each attribute must be a NOUN or short NOUN-PHRASE (1-3 words, singular form).  
• Ignore neutral instructions, usage tips, or negative remarks.  
• Ignore vague verbs/adjectives like “works great”, “serves the purpose”.  
• Remove duplicates.  
• Respond with the JSON object only – no extra text, no code fences.

EXAMPLES  

1️⃣  
Review:  
“I love the **fragrance** and how **long-lasting** it is. Totally **aluminum-free** and **gentle on skin**.”  
Output:  
{"attributes": ["Fragrance", "Long-lasting durability", "Aluminum-free", "Gentle on skin"]}

2️⃣  
Review (mixed sentiment):  
“The deodorant smells nice but stains my shirts.”  
Output:  
{"attributes": ["Pleasant scent"]}

3️⃣  
Negative review only:  
“Leaves yellow stains and the scent is awful.”  
Output:  
{"attributes": []} """

async def _extract_one(text: str, retry: int = 3) -> List[str]:
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]

    for attempt in range(1, retry + 1):
        try:
            resp = await openai_client_async.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                response_format={"type": "json_object"},  # ensures valid JSON
                temperature=0.0,
                max_tokens=120,
            )
            payload = json.loads(resp.choices[0].message.content)
            attrs = payload.get("attributes", [])
            return [a.strip() for a in attrs if isinstance(a, str) and a.strip()]
        except Exception as exc:
            logging.warning("[extract attempt %d] %s", attempt, exc)
            await asyncio.sleep(2 ** attempt)  # exponential back‑off
    return []

# ---------------- fallback rule‑based ----------------

def _fallback_extraction(text: str) -> List[str]:
    doc = _nlp(text)
    found: set[str] = set()
    for sent in doc.sents:
        lower = sent.text.lower()
        for kw in _FALLBACK_KEYWORDS:
            if kw in lower:
                found.add(kw.capitalize())
    return sorted(found)

# ---------------- batch API ----------------

_MAX_CONCURRENCY = 20

async def _process_review(review: Dict[str, Any]):
    text = review.get("body", "")
    attrs = await _extract_one(text) if text else []
    if not attrs:
        attrs = _fallback_extraction(text)
    review["delight_attributes"] = attrs
    return review

async def _run_async_pool(reviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sem = asyncio.Semaphore(_MAX_CONCURRENCY)

    async def guard(r):
        async with sem:
            return await _process_review(r.copy())

    return await asyncio.gather(*(guard(r) for r in reviews))


def extract_attributes_batch(reviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Synchronous wrapper for callers (CLI)."""
    return asyncio.run(_run_async_pool(reviews))