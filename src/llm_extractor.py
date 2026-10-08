import os
import json
import re
import time
import requests

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "openai/gpt-oss-20b"

POLLINATIONS_KEY = os.getenv("POLLINATIONS_KEY", "")
POLLINATIONS_URL = "https://gen.pollinations.ai/v1/chat/completions"
POLLINATIONS_MODEL = "openai"

GROCERY_PROMPT = """You are a grocery basket extractor. Convert user text to JSON.
Return ONLY valid JSON. No explanation. No markdown.
Schema:
{"domain":"grocery","items":[{"raw":string,"normalized":string,"quantity":string|null,"category":string|null}],"intent":"new_purchase"|"refill"|null,"urgency":"low"|"medium"|"high","language":string,"confidence":float}
Rules:
- Split comma/and-separated items.
- Refill words -> intent="refill".
- Urgency words -> urgency="high".
- Never invent items.
User: __USER_TEXT__
"""

PHARMACY_PROMPT = """You are a pharmacy basket extractor. Convert user text to JSON.
Return ONLY valid JSON. No explanation. No markdown.
Schema:
{"domain":"pharmacy","items":[{"raw":string,"normalized":string,"bnf_chapter":string,"strength":string|null,"form":string|null}],"intent":"refill"|"new_prescription"|null,"urgency":"low"|"medium"|"high","language":string,"confidence":float}
BNF: 01 GI, 02 Cardio, 03 Resp, 04 CNS, 05 Inf, 06 Endo, 07 GU, 08 Malig, 09 Nutr, 10 MSK, 11 Eye, 12 ENT, 13 Skin.
Rules:
- Refill words -> intent="refill".
- Never invent drugs.
User: __USER_TEXT__
"""

def call_groq(prompt, timeout=60, max_retries=3):
    if not GROQ_API_KEY or "YOUR_GROQ" in GROQ_API_KEY:
        return None, "groq_not_configured"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }
    body = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "Return valid JSON only. No explanation."},
            {"role": "user", "content": prompt}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.1
    }
    delays = [2, 5, 10]
    for attempt in range(max_retries):
        try:
            r = requests.post(GROQ_URL, headers=headers, json=body, timeout=timeout)
            if r.status_code == 200:
                data = r.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if content and content.strip():
                    return content, f"groq (attempt {attempt + 1})"
            print(f"  groq attempt {attempt + 1}: status={r.status_code}, body={r.text[:100]}")
            time.sleep(delays[min(attempt, len(delays) - 1)])
        except Exception as e:
            print(f"  groq attempt {attempt + 1}: {e}")
            time.sleep(delays[min(attempt, len(delays) - 1)])
    return None, "groq_failed"

def call_pollinations(prompt, timeout=90, max_retries=3):
    headers = {"Content-Type": "application/json"}
    body = {
        "model": POLLINATIONS_MODEL,
        "messages": [
            {"role": "system", "content": "Return valid JSON only. No explanation."},
            {"role": "user", "content": prompt}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.1
    }
    delays = [3, 8, 15]
    for attempt in range(max_retries):
        try:
            r = requests.post(POLLINATIONS_URL, headers=headers, json=body, timeout=timeout)
            if r.status_code == 200:
                data = r.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if content and content.strip():
                    return content, f"pollinations (attempt {attempt + 1})"
            print(f"  pollinations attempt {attempt + 1}: status={r.status_code}, body={r.text[:100]}")
            time.sleep(delays[min(attempt, len(delays) - 1)])
        except Exception as e:
            print(f"  pollinations attempt {attempt + 1}: {e}")
            time.sleep(delays[min(attempt, len(delays) - 1)])
    return None, "pollinations_failed"

def call_llm(prompt):
    print("Trying Groq...")
    raw, provider = call_groq(prompt)
    if raw:
        return raw, provider

    print("Groq failed, falling back to Pollinations...")
    raw, provider = call_pollinations(prompt)
    if raw:
        return raw, provider

    return None, "all_failed"

def parse_json_from_text(text):
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None

def validate_extraction(data, expected_domain):
    if not isinstance(data, dict):
        return False, "not a dict"
    if data.get("domain") != expected_domain:
        return False, f"wrong domain: {data.get('domain')}"
    if not isinstance(data.get("items"), list):
        return False, "items not a list"
    return True, "ok"

def extract_from_text(user_text, domain):
    result = {"status": "failed", "data": None, "message": "", "provider": None, "fallback": True}

    if domain == "grocery":
        prompt = GROCERY_PROMPT.replace("__USER_TEXT__", user_text)
    elif domain == "pharmacy":
        prompt = PHARMACY_PROMPT.replace("__USER_TEXT__", user_text)
    else:
        result["message"] = f"Unknown domain: {domain}"
        return result

    raw, provider = call_llm(prompt)
    result["provider"] = provider

    if raw is None:
        result["message"] = "All LLM providers failed — please use manual input"
        return result

    parsed = parse_json_from_text(raw)
    if parsed is None:
        result["message"] = f"{provider}: output not valid JSON"
        return result

    ok, reason = validate_extraction(parsed, domain)
    if not ok:
        result["message"] = f"Validation failed: {reason}"
        return result

    if len(parsed.get("items", [])) == 0:
        result["message"] = "No items extracted — try manual input"
        return result

    result["status"] = "success"
    result["data"] = parsed
    result["message"] = f"Extraction successful ({provider})"
    result["fallback"] = False
    return result
