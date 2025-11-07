# # # # # import streamlit as st
# # # # # from azure_blob import upload_to_blob
# # # # # from extract_text import extract_text_from_bytes
# # # # # from llm_analyzer import analyze_resume_with_llm

# # # # # st.title("🧠 Azure Resume ↔ JD Text Extractor")

# # # # # uploaded_file = st.file_uploader("Upload Resume or JD (PDF/DOCX)", type=["pdf", "docx"])

# # # # # if uploaded_file:
# # # # #     with st.spinner("Uploading and extracting text..."):
# # # # #         upload_to_blob(uploaded_file.getvalue(), uploaded_file.name)
# # # # #         text = extract_text_from_bytes(uploaded_file.getvalue())

# # # # #     st.success("✅ Text extracted successfully!")
# # # # #     st.text_area("Extracted Text", text[:3000], height=400)
# # # # # if st.button("🔍 Analyze with AI"):
# # # # #     with st.spinner("Analyzing resume vs job description using AI..."):
# # # # #         ai_output = analyze_resume_with_llm(resume_text, jd_text)

# # # # #     st.subheader("💡 AI Insights & Suggestions")
# # # # #     st.write(ai_output)

# # # # import streamlit as st
# # # # from azure_blob import upload_to_blob
# # # # from extract_text import extract_text_from_bytes
# # # # from llm_analyzer import analyze_resume_with_llm
# # # # from similarity_analyzer import calculate_similarity, extract_skills
# # # # import plotly.express as px

# # # # st.set_page_config(page_title="AI Resume ↔ JD Analyzer", layout="wide")
# # # # st.title("🤖 AI Resume ↔ JD Analyzer")

# # # # # --- Upload both files ---
# # # # col1, col2 = st.columns(2)
# # # # with col1:
# # # #     resume_file = st.file_uploader("📄 Upload Resume", type=["pdf", "docx"])
# # # # with col2:
# # # #     jd_file = st.file_uploader("🧾 Upload Job Description", type=["pdf", "docx", "txt"])

# # # # # --- Continue only if both are uploaded ---
# # # # if resume_file and jd_file:
# # # #     with st.spinner("Uploading and extracting text..."):
# # # #         resume_text = extract_text_from_bytes(resume_file.getvalue())
# # # #         jd_text = extract_text_from_bytes(jd_file.getvalue())
# # # #         upload_to_blob(resume_file.getvalue(), resume_file.name)
# # # #         upload_to_blob(jd_file.getvalue(), jd_file.name)

# # # #     st.success("✅ Both files processed successfully!")
# # # #     st.subheader("🧠 Extracted Resume Text")
# # # #     st.text_area("Resume", resume_text[:3000], height=200)
# # # #     st.subheader("📜 Extracted Job Description Text")
# # # #     st.text_area("Job Description", jd_text[:3000], height=200)

# # # #     # === AI Comparison Section ===
# # # #     if st.button("💡 Analyze Resume with Azure GPT-4o-mini"):
# # # #         with st.spinner("Analyzing Resume ↔ JD using Azure AI..."):
# # # #             raw_output = analyze_resume_with_llm(resume_text, jd_text)
    
# # # #     import json

# # # # st.subheader("💡 LLM Suggestions")

# # # # try:
# # # #     data = json.loads(raw_output)  # parse the JSON string into a Python dict
    
# # # #     # --- Section 1: Matched Skills ---
# # # #     st.markdown("### ✅ Matched Skills")
# # # #     if data.get("Matched_Skills"):
# # # #         st.success(", ".join(data["Matched_Skills"]))
# # # #     else:
# # # #         st.info("No matched skills found.")

# # # #     # --- Section 2: Missing Skills ---
# # # #     st.markdown("### ❌ Missing Skills")
# # # #     if data.get("Missing_Skills"):
# # # #         for skill in data["Missing_Skills"]:
# # # #             st.warning(f"• {skill}")
# # # #     else:
# # # #         st.info("No missing skills detected.")

# # # #     # --- Section 3: Suggestions ---
# # # #     st.markdown("### 🧭 Suggestions for Improvement")
# # # #     if data.get("Suggestions"):
# # # #         for suggestion in data["Suggestions"]:
# # # #             st.markdown(f"- {suggestion}")
# # # #     else:
# # # #         st.info("No improvement suggestions available.")

# # # # except Exception:
# # # #     st.error("⚠️ Could not parse LLM output. Here's the raw response:")
# # # #     st.write(raw_output)



# # # #     # === Skill Match + Similarity Section ===
# # # #     with st.spinner("Calculating similarity..."):
# # # #         similarity_score = calculate_similarity(resume_text, jd_text)
# # # #         resume_skills = extract_skills(resume_text)
# # # #         jd_skills = extract_skills(jd_text)
# # # #         matched = list(set(resume_skills) & set(jd_skills))
# # # #         missing = list(set(jd_skills) - set(resume_skills))

# # # #     st.metric("Resume ↔ JD Similarity", f"{similarity_score}%")
# # # #     st.write(f"✅ **Matched Skills:** {', '.join(matched) if matched else 'None'}")
# # # #     st.write(f"❌ **Missing Skills:** {', '.join(missing) if missing else 'None'}")

# # # #     if jd_skills:
# # # #         data = {"Skill Type": ["Matched"] * len(matched) + ["Missing"] * len(missing),
# # # #                 "Skill": matched + missing}
# # # #         if data["Skill"]:
# # # #             fig = px.bar(data, x="Skill", color="Skill Type",
# # # #                          title="Matched vs Missing Skills", text_auto=True)
# # # #             st.plotly_chart(fig, use_container_width=True)
# # # # else:
# # # #     st.info("⬆️ Please upload both Resume and Job Description files to begin.")


# # # import streamlit as st
# # # import json
# # # import numpy as np
# # # import plotly.express as px
# # # from azure_blob import upload_to_blob
# # # from extract_text import extract_text_from_bytes
# # # from llm_analyzer import analyze_resume_with_llm
# # # from openai import AzureOpenAI
# # # from azure_config import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY

# # # # --- Streamlit Page Setup ---
# # # st.set_page_config(page_title="AI Resume ↔ JD Analyzer", layout="wide")
# # # st.title("🤖 AI Resume ↔ JD Analyzer (Powered by Azure OpenAI)")

# # # # --- Initialize Azure OpenAI Client (for embeddings) ---
# # # embedding_client = AzureOpenAI(
# # #     api_version="2024-12-01-preview",
# # #     azure_endpoint=AZURE_OPENAI_ENDPOINT,
# # #     api_key=AZURE_OPENAI_KEY,
# # # )

# # # # --- Utility Functions ---
# # # def cosine_similarity(a, b):
# # #     return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# # # def calculate_similarity(resume_text: str, jd_text: str) -> float:
# # #     """Generate embeddings for both texts and compute cosine similarity."""
# # #     try:
# # #         resume_embed = embedding_client.embeddings.create(
# # #             model="text-embedding-3-small",
# # #             input=str(resume_text)
# # #         ).data[0].embedding

# # #         jd_embed = embedding_client.embeddings.create(
# # #             model="text-embedding-3-small",
# # #             input=str(jd_text)
# # #         ).data[0].embedding

# # #         score = cosine_similarity(np.array(resume_embed), np.array(jd_embed))
# # #         return round(score * 100, 2)
# # #     except Exception as e:
# # #         st.error(f"⚠️ Azure Embedding Error: {e}")
# # #         return 0.0


# # # def extract_skills(text: str):
# # #     """Very simple skill keyword extraction."""
# # #     skill_keywords = [
# # #         "python", "java", "c++", "sql", "html", "css", "react", "azure",
# # #         "machine learning", "deep learning", "docker", "api", "javascript",
# # #         "mongodb", "streamlit", "langchain", "rag", "nosql"
# # #     ]
# # #     found = [s for s in skill_keywords if s.lower() in text.lower()]
# # #     return list(set(found))


# # # # --- Upload both files ---
# # # col1, col2 = st.columns(2)
# # # with col1:
# # #     resume_file = st.file_uploader("📄 Upload Resume", type=["pdf", "docx"])
# # # with col2:
# # #     jd_file = st.file_uploader("🧾 Upload Job Description", type=["pdf", "docx", "txt"])

# # # # --- Continue only if both files uploaded ---
# # # if resume_file and jd_file:
# # #     with st.spinner("📤 Uploading and extracting text from files..."):
# # #         resume_text = extract_text_from_bytes(resume_file.getvalue())
# # #         jd_text = extract_text_from_bytes(jd_file.getvalue())
# # #         upload_to_blob(resume_file.getvalue(), resume_file.name)
# # #         upload_to_blob(jd_file.getvalue(), jd_file.name)

# # #     st.success("✅ Files processed successfully!")

# # #     with st.expander("🧠 Extracted Resume Text", expanded=False):
# # #         st.text_area("Resume", resume_text[:3000], height=200)
# # #     with st.expander("📜 Extracted Job Description Text", expanded=False):
# # #         st.text_area("Job Description", jd_text[:3000], height=200)

# # #     # --- Analyze Button ---
# # #     if st.button("💡 Analyze Resume with Azure GPT-4o-mini"):
# # #         with st.spinner("Analyzing Resume ↔ JD using Azure AI..."):
# # #             raw_output = analyze_resume_with_llm(resume_text, jd_text)

# # #         st.subheader("💡 LLM Suggestions")

# # #         # --- Parse JSON safely ---
# # #         try:
# # #             data = json.loads(raw_output)

# # #             # === Section 1: Matched Skills ===
# # #             st.markdown("### ✅ Matched Skills")
# # #             matched_skills = data.get("Matched_Skills", [])
# # #             if matched_skills:
# # #                 st.success(", ".join(matched_skills))
# # #             else:
# # #                 st.info("No matched skills found.")

# # #             # === Section 2: Missing Skills ===
# # #             st.markdown("### ❌ Missing Skills")
# # #             missing_skills = data.get("Missing_Skills", [])
# # #             if missing_skills:
# # #                 for skill in missing_skills:
# # #                     st.warning(f"• {skill}")
# # #             else:
# # #                 st.info("No missing skills detected.")

# # #             # === Section 3: Suggestions ===
# # #             st.markdown("### 🧭 Suggestions for Improvement")
# # #             suggestions = data.get("Suggestions", [])
# # #             if suggestions:
# # #                 for s in suggestions:
# # #                     st.markdown(f"- {s}")
# # #             else:
# # #                 st.info("No suggestions available.")

# # #             # === Similarity Calculation ===
# # #             with st.spinner("📊 Calculating similarity..."):
# # #                 similarity_score = calculate_similarity(resume_text, jd_text)
# # #                 resume_skills = extract_skills(resume_text)
# # #                 jd_skills = extract_skills(jd_text)
# # #                 matched = list(set(resume_skills) & set(jd_skills))
# # #                 missing = list(set(jd_skills) - set(resume_skills))

# # #             st.metric("Resume ↔ JD Similarity", f"{similarity_score}%")
# # #             st.write(f"✅ **Matched Skills:** {', '.join(matched) if matched else 'None'}")
# # #             st.write(f"❌ **Missing Skills:** {', '.join(missing) if missing else 'None'}")

# # #             # === Skill Visualization ===
# # #             if jd_skills:
# # #                 data_chart = {
# # #                     "Skill Type": ["Matched"] * len(matched) + ["Missing"] * len(missing),
# # #                     "Skill": matched + missing,
# # #                 }
# # #                 if data_chart["Skill"]:
# # #                     fig = px.bar(
# # #                         data_chart,
# # #                         x="Skill",
# # #                         color="Skill Type",
# # #                         title="Matched vs Missing Skills",
# # #                         text_auto=True,
# # #                         color_discrete_map={"Matched": "green", "Missing": "red"},
# # #                     )
# # #                     st.plotly_chart(fig, use_container_width=True)

# # #         except Exception as e:
# # #             st.error(f"⚠️ Could not parse LLM output. Showing raw response:\n\n{e}")
# # #             st.code(raw_output, language="json")

# # # else:
# # #     st.info("⬆️ Please upload both Resume and Job Description files to begin.")



# # import os
# # import json
# # import numpy as np
# # import plotly.express as px
# # import streamlit as st
# # from openai import AzureOpenAI
# # from azure_config import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY
# # from azure_blob import upload_to_blob
# # from extract_text import extract_text_from_bytes
# # from llm_analyzer import analyze_resume_with_llm

# # # =========================
# # # Streamlit Setup
# # # =========================
# # st.set_page_config(page_title="AI Resume ↔ JD Analyzer", layout="wide")
# # st.title("🤖 AI Resume & JD Analyzer (Powered by Azure OpenAI)")

# # # Sidebar Navigation
# # mode = st.sidebar.radio(
# #     "🔍 Select Mode",
# #     ["Single Resume ↔ JD Analyzer", "Multi-Resume Ranking"]
# # )

# # # =========================
# # # Initialize Azure Client
# # # =========================
# # client = AzureOpenAI(
# #     api_version="2024-12-01-preview",
# #     azure_endpoint=AZURE_OPENAI_ENDPOINT,
# #     api_key=AZURE_OPENAI_KEY,
# # )

# # # =========================
# # # Helper Functions
# # # =========================
# # def cosine_similarity(a, b):
# #     a, b = np.array(a, dtype=float), np.array(b, dtype=float)
# #     return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# # def get_text_embedding(text):
# #     """Generate Azure text embedding."""
# #     if not text or not isinstance(text, str):
# #         raise ValueError("Input text must be a valid non-empty string.")
# #     resp = client.embeddings.create(
# #         model="text-embedding-3-small",
# #         input=[text.strip()]
# #     )
# #     return resp.data[0].embedding


# # def calculate_similarity(resume_text, jd_text):
# #     """Compute cosine similarity between resume & JD."""
# #     try:
# #         r_embed = get_text_embedding(resume_text)
# #         j_embed = get_text_embedding(jd_text)
# #         return round(cosine_similarity(r_embed, j_embed) * 100, 2)
# #     except Exception as e:
# #         st.error(f"⚠️ Azure Embedding Error: {e}")
# #         return 0.0


# # def extract_skills(text):
# #     """Keyword-based skill extractor."""
# #     skills = [
# #         "python", "java", "c++", "sql", "html", "css", "react", "javascript",
# #         "azure", "docker", "api", "rest", "machine learning", "deep learning",
# #         "mongodb", "nosql", "langchain", "rag", "streamlit", "gradio"
# #     ]
# #     found = [s for s in skills if s.lower() in text.lower()]
# #     return list(set(found))

# # # =========================
# # # MODE 1 — Single Resume ↔ JD
# # # =========================
# # if mode == "Single Resume ↔ JD Analyzer":
# #     col1, col2 = st.columns(2)
# #     with col1:
# #         resume_file = st.file_uploader("📄 Upload Resume", type=["pdf", "docx"])
# #     with col2:
# #         jd_file = st.file_uploader("🧾 Upload Job Description", type=["pdf", "docx", "txt"])

# #     if resume_file and jd_file:
# #         with st.spinner("📤 Uploading and extracting text..."):
# #             resume_text = extract_text_from_bytes(resume_file.getvalue())
# #             jd_text = extract_text_from_bytes(jd_file.getvalue())

# #             upload_to_blob(resume_file.getvalue(), resume_file.name)
# #             upload_to_blob(jd_file.getvalue(), jd_file.name)

# #         st.success("✅ Files processed successfully!")

# #         with st.expander("🧠 Extracted Resume Text", expanded=False):
# #             st.text_area("Resume", resume_text[:3000], height=200)
# #         with st.expander("📜 Extracted Job Description Text", expanded=False):
# #             st.text_area("Job Description", jd_text[:3000], height=200)

# #         if st.button("💡 Analyze Resume with Azure GPT-4o-mini"):
# #             with st.spinner("Analyzing Resume ↔ JD using Azure AI..."):
# #                 raw_output = analyze_resume_with_llm(resume_text, jd_text)

# #             st.subheader("💡 LLM Suggestions")

# #             try:
# #                 data = json.loads(raw_output)

# #                 # Matched Skills
# #                 st.markdown("### ✅ Matched Skills")
# #                 matched = data.get("Matched_Skills", [])
# #                 if matched:
# #                     st.success(", ".join(matched))
# #                 else:
# #                     st.info("No matched skills found.")

# #                 # Missing Skills
# #                 st.markdown("### ❌ Missing Skills")
# #                 missing = data.get("Missing_Skills", [])
# #                 if missing:
# #                     for m in missing:
# #                         st.warning(f"• {m}")
# #                 else:
# #                     st.info("No missing skills detected.")

# #                 # Suggestions
# #                 st.markdown("### 🧭 Suggestions for Improvement")
# #                 for s in data.get("Suggestions", []):
# #                     st.markdown(f"- {s}")

# #             except Exception as e:
# #                 st.error(f"⚠️ Could not parse LLM output: {e}")
# #                 st.code(raw_output, language="json")

# #             # === Similarity & Visualization ===
# #             with st.spinner("📊 Calculating similarity..."):
# #                 similarity = calculate_similarity(resume_text, jd_text)
# #                 resume_skills = extract_skills(resume_text)
# #                 jd_skills = extract_skills(jd_text)
# #                 matched_skills = list(set(resume_skills) & set(jd_skills))
# #                 missing_skills = list(set(jd_skills) - set(resume_skills))

# #             st.metric("Resume ↔ JD Similarity", f"{similarity}%")
# #             st.write(f"✅ **Matched Skills:** {', '.join(matched_skills) if matched_skills else 'None'}")
# #             st.write(f"❌ **Missing Skills:** {', '.join(missing_skills) if missing_skills else 'None'}")

# #             if jd_skills:
# #                 chart_data = {
# #                     "Skill Type": ["Matched"] * len(matched_skills) + ["Missing"] * len(missing_skills),
# #                     "Skill": matched_skills + missing_skills,
# #                 }
# #                 fig = px.bar(
# #                     chart_data,
# #                     x="Skill",
# #                     color="Skill Type",
# #                     title="Matched vs Missing Skills",
# #                     color_discrete_map={"Matched": "green", "Missing": "red"},
# #                 )
# #                 st.plotly_chart(fig, use_container_width=True)

# #     else:
# #         st.info("⬆️ Please upload both Resume and Job Description files to begin.")

# # # =========================
# # # MODE 2 — Multi-Resume Ranking
# # # =========================
# # elif mode == "Multi-Resume Ranking":
# #     st.subheader("📊 Compare Multiple Resumes Against One Job Description")
# #     jd_file = st.file_uploader("🧾 Upload Job Description", type=["pdf", "docx", "txt"])
# #     resume_files = st.file_uploader("📄 Upload Multiple Resumes", type=["pdf", "docx"], accept_multiple_files=True)

# #     if jd_file and resume_files:
# #         with st.spinner("Extracting JD text..."):
# #             jd_text = extract_text_from_bytes(jd_file.getvalue())

# #         results = []
# #         with st.spinner("Calculating similarity for each resume..."):
# #             for resume_file in resume_files:
# #                 resume_text = extract_text_from_bytes(resume_file.getvalue())
# #                 upload_to_blob(resume_file.getvalue(), resume_file.name)
# #                 sim = calculate_similarity(resume_text, jd_text)
# #                 results.append((resume_file.name, sim))

# #         # Sort results descending
# #         results.sort(key=lambda x: x[1], reverse=True)

# #         # Display results
# #         st.success("✅ Ranking Complete!")
# #         st.subheader("🏆 Resume Ranking by Similarity")

# #         for i, (name, score) in enumerate(results, 1):
# #             st.write(f"**{i}. {name}** — {score}% match")

# #         # Chart visualization
# #         names = [r[0] for r in results]
# #         scores = [r[1] for r in results]
# #         fig = px.bar(
# #             x=names,
# #             y=scores,
# #             title="Resume Similarity Scores",
# #             labels={"x": "Resume", "y": "Similarity (%)"},
# #             text=[f"{s}%" for s in scores],
# #             color=scores,
# #             color_continuous_scale="greens",
# #         )
# #         st.plotly_chart(fig, use_container_width=True)
# #     else:
# #         st.info("⬆️ Upload one JD and multiple resumes to start ranking.")


# # ==========================================
# # app.py
# # ==========================================
# # Azure AI Resume ↔ JD Analyzer
# # Uses GPT-4o-mini for insights and text-embedding-3-small for similarity
# # ==========================================

# import streamlit as st
# import json
# import plotly.express as px
# from azure_blob import upload_to_blob
# from extract_text import extract_text_from_bytes
# from llm_analyzer import analyze_resume_with_llm
# from similarity_analyzer import calculate_similarity, extract_skills

# # ---- Streamlit Configuration ----
# st.set_page_config(page_title="AI Resume ↔ JD Analyzer", layout="wide")
# st.title("🤖 AI Resume ↔ JD Analyzer")

# # ---- File Upload ----
# col1, col2 = st.columns(2)
# with col1:
#     resume_file = st.file_uploader("📄 Upload Resume", type=["pdf", "docx"])
# with col2:
#     jd_file = st.file_uploader("🧾 Upload Job Description", type=["pdf", "docx", "txt"])

# # ---- Main Processing ----
# if resume_file and jd_file:
#     with st.spinner("⏳ Uploading and extracting text..."):
#         resume_text = extract_text_from_bytes(resume_file.getvalue())
#         jd_text = extract_text_from_bytes(jd_file.getvalue())
#         upload_to_blob(resume_file.getvalue(), resume_file.name)
#         upload_to_blob(jd_file.getvalue(), jd_file.name)

#     st.success("✅ Files processed successfully!")

#     st.subheader("🧠 Extracted Resume Text")
#     st.text_area("Resume", resume_text[:3000], height=200)

#     st.subheader("📜 Extracted Job Description Text")
#     st.text_area("Job Description", jd_text[:3000], height=200)

#     # ---- Analysis Section ----
#     if st.button("💡 Analyze Resume with Azure GPT-4o-mini"):
#         with st.spinner("🤔 Analyzing Resume ↔ JD using Azure LLM..."):
#             raw_output = analyze_resume_with_llm(resume_text, jd_text)

#         st.subheader("💡 LLM Suggestions")

#         try:
#             data = json.loads(raw_output)

#             st.markdown("### ✅ Matched Skills")
#             if data.get("Matched_Skills"):
#                 st.success(", ".join(data["Matched_Skills"]))
#             else:
#                 st.info("No matched skills found.")

#             st.markdown("### ❌ Missing Skills")
#             if data.get("Missing_Skills"):
#                 for skill in data["Missing_Skills"]:
#                     st.warning(f"• {skill}")
#             else:
#                 st.info("No missing skills detected.")

#             st.markdown("### 🧭 Suggestions for Improvement")
#             if data.get("Suggestions"):
#                 for suggestion in data["Suggestions"]:
#                     st.markdown(f"- {suggestion}")
#             else:
#                 st.info("No improvement suggestions available.")

#         except Exception:
#             st.error("⚠️ Could not parse LLM output. Here's the raw response:")
#             st.write(raw_output)

#         # ---- Similarity Section ----
#         with st.spinner("📊 Calculating similarity..."):
#             similarity_score = calculate_similarity(resume_text, jd_text)
#             resume_skills = extract_skills(resume_text)
#             jd_skills = extract_skills(jd_text)
#             matched = list(set(resume_skills) & set(jd_skills))
#             missing = list(set(jd_skills) - set(resume_skills))

#         st.metric("Resume ↔ JD Similarity", f"{similarity_score}%")
#         st.write(f"✅ **Matched Skills:** {', '.join(matched) if matched else 'None'}")
#         st.write(f"❌ **Missing Skills:** {', '.join(missing) if missing else 'None'}")

#         if jd_skills:
#             chart_data = {
#                 "Skill Type": ["Matched"] * len(matched) + ["Missing"] * len(missing),
#                 "Skill": matched + missing,
#             }

#             if chart_data["Skill"]:
#                 fig = px.bar(
#                     chart_data,
#                     x="Skill",
#                     color="Skill Type",
#                     title="Matched vs Missing Skills",
#                     text_auto=True,
#                 )
#                 st.plotly_chart(fig, use_container_width=True)
# else:
#     st.info("⬆️ Please upload both Resume and Job Description files to begin.")


# ==========================================
# app.py
# ==========================================

import streamlit as st
import pandas as pd
from extract_text import extract_text_from_bytes
from llm_analyzer import analyze_resume_with_llm
from similarity_analyzer import calculate_similarity, rank_resumes

st.set_page_config(page_title="Azure AI Resume Analyzer", layout="wide")
st.sidebar.title("🧭 Navigation")
mode = st.sidebar.radio("Select Mode", ["👨‍💻 Candidate", "🧑‍💼 Recruiter"])

# --------------------------------------------------
# 👨‍💻 Candidate Mode — Analyze one Resume vs JD
# --------------------------------------------------
if mode == "👨‍💻 Candidate":
    st.title("👨‍💻 Candidate Mode — Analyze Resume ↔ JD")
    col1, col2 = st.columns(2)
    with col1:
        resume_file = st.file_uploader("📄 Upload Resume", type=["pdf", "docx"])
    with col2:
        jd_file = st.file_uploader("🧾 Upload Job Description", type=["pdf", "docx", "txt"])

    if resume_file and jd_file:
        with st.spinner("Extracting and analyzing text..."):
            resume_text = extract_text_from_bytes(resume_file.getvalue())
            jd_text = extract_text_from_bytes(jd_file.getvalue())

        st.success("✅ Text extracted successfully.")
        st.subheader("Resume Text")
        st.text_area("Resume", resume_text[:2000], height=200)
        st.subheader("Job Description")
        st.text_area("JD", jd_text[:2000], height=200)

        if st.button("💡 Analyze with Azure AI"):
            with st.spinner("Running GPT-4o-mini analysis..."):
                llm_output = analyze_resume_with_llm(resume_text, jd_text)
            st.write("### 🤖 LLM Suggestions")
            st.json(llm_output)

            similarity = calculate_similarity(resume_text, jd_text)
            st.metric("Resume ↔ JD Similarity", f"{similarity}%")

# --------------------------------------------------
# 🧑‍💼 Recruiter Mode — Rank Multiple Resumes
# --------------------------------------------------
elif mode == "🧑‍💼 Recruiter":
    st.title("🧑‍💼 Recruiter Mode — Rank Multiple Resumes")
    jd_file = st.file_uploader("🧾 Upload Job Description", type=["pdf", "docx", "txt"])
    resume_files = st.file_uploader("📄 Upload Multiple Resumes", type=["pdf", "docx"], accept_multiple_files=True)

    if jd_file and resume_files:
        with st.spinner("Extracting texts..."):
            jd_text = extract_text_from_bytes(jd_file.getvalue())
            resumes = {f.name: extract_text_from_bytes(f.getvalue()) for f in resume_files}

        st.info(f"✅ {len(resumes)} resumes uploaded successfully.")

        if st.button("🏁 Rank Candidates"):
            with st.spinner("Analyzing and ranking resumes..."):
                results = rank_resumes(resumes, jd_text)

            st.success("✅ Ranking complete!")

            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)

            best = df.iloc[0]
            st.markdown(f"🏆 **Top Candidate:** {best['Resume']} — {best['Adjusted_Score']}%")
            st.caption(best["Reason"])
    else:
        st.info("⬆️ Upload a Job Description and multiple resumes to begin ranking.")
