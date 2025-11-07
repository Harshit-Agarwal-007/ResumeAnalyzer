# ==========================================
# similarity_analyzer.py
# ==========================================

import numpy as np
import json
from openai import AzureOpenAI
from azure_config import (
    AZURE_EMBEDDING_ENDPOINT,
    AZURE_EMBEDDING_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_KEY,
)

# ---- Initialize Azure Clients ----
EMBEDDING_DEPLOYMENT_NAME = "text-embedding-3-small"   # your embedding model deployment
CHAT_DEPLOYMENT_NAME = "gpt-4o-mini-rja"               # your GPT deployment name

embedding_client = AzureOpenAI(
    api_key=AZURE_EMBEDDING_KEY,
    azure_endpoint=AZURE_EMBEDDING_ENDPOINT,
    api_version="2024-12-01-preview"
)

chat_client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version="2024-12-01-preview"
)

# ---------- Embeddings ----------
def get_embedding(text: str):
    if not isinstance(text, str) or not text.strip():
        return [0.0] * 1536
    response = embedding_client.embeddings.create(
        model=EMBEDDING_DEPLOYMENT_NAME,
        input=[text.strip()]
    )
    return response.data[0].embedding

def cosine_similarity(a, b):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def calculate_similarity(resume_text, jd_text):
    e1 = get_embedding(resume_text)
    e2 = get_embedding(jd_text)
    return round(cosine_similarity(e1, e2) * 100, 2)

# ---------- Context-Aware Ranking ----------
def context_aware_adjustment(resume_text, jd_text, base_score):
    prompt = f"""
    You are an expert recruiter. Evaluate this RESUME for the JOB DESCRIPTION.
    Base embedding similarity: {base_score}%.

    Consider:
    - Role relevance
    - Experience level
    - Project/tech alignment
    - Communication and clarity

    Respond only in JSON:
    {{
      "Adjusted_Score": <number>,
      "Reason": "<short explanation>"
    }}

    RESUME:
    {resume_text[:2000]}

    JD:
    {jd_text[:2000]}
    """

    try:
        resp = chat_client.chat.completions.create(
            model=CHAT_DEPLOYMENT_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.3
        )
        data = json.loads(resp.choices[0].message.content)
        return data.get("Adjusted_Score", base_score), data.get("Reason", "No reason provided.")
    except Exception as e:
        print(f"⚠️ Context adjustment failed: {e}")
        return base_score, "Used base embedding similarity."

def rank_resumes(resume_texts, jd_text):
    """Compute and rank multiple resumes."""
    results = []
    for name, text in resume_texts.items():
        base = calculate_similarity(text, jd_text)
        adjusted, reason = context_aware_adjustment(text, jd_text, base)
        results.append({
            "Resume": name,
            "Base_Similarity": base,
            "Adjusted_Score": adjusted,
            "Reason": reason
        })
    return sorted(results, key=lambda x: x["Adjusted_Score"], reverse=True)
