import os
from openai import AzureOpenAI
from azure_config import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY
# ---- Azure configuration (match your working script) ----
endpoint = AZURE_OPENAI_ENDPOINT
api_key = AZURE_OPENAI_KEY

model_name = "gpt-4o-mini"
deployment = "gpt-4o-mini-rja"

api_version = "2024-12-01-preview"
# ---- Initialize client ----
client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=api_key,
)

# ---- Main function ----
def analyze_resume_with_llm(resume_text, jd_text):
    """
    Compare resume and job description using Azure GPT-4o-mini.
    Returns structured JSON-like text with Matched Skills, Missing Skills, and Suggestions.
    """
    system_prompt = """You are an expert AI Resume Analyzer.
Compare the RESUME and JOB DESCRIPTION carefully and always respond *only* in this JSON format:
{
  "Matched_Skills": [],
  "Missing_Skills": [],
  "Suggestions": []
}"""

    user_prompt = f"""
RESUME:
{resume_text}

JOB DESCRIPTION:
{jd_text}
"""

    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=700,
            temperature=0.4,
            top_p=1.0,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"❌ Azure LLM Error: {e}"
