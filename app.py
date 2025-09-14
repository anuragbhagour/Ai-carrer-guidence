# app.py (relevant excerpt)
import streamlit as st
from structured_resume import structured_resume
from build_kb import load_knowledge_base, build_to_index
from Query import query_kb
from rag_pipeline import generate_guidance

# Load KB once
kb = load_knowledge_base()
model, index, embeddings = build_to_index(kb)

st.title("AI Career Guidance Bot")

uploaded_file = st.file_uploader("Upload your Resume (PDF)", type="pdf")

if uploaded_file is not None:
    resume_data = structured_resume(uploaded_file)
    skills_list = resume_data.get("Extracted_Skills", [])
    skills_str = ", ".join(skills_list)

    if skills_list:
        # Retrieval
        top_k = 3
        retrieved_indices = query_kb(skills_str, model, index, kb, top_k=top_k, return_docs=True)
        # NOTE: adjust query_kb to optionally return doc dicts (title+desc). See below.

        # If query_kb returns titles only, instead build a retrieved_docs list:
        # retrieved_docs = [kb[i] for i in retrieved_indices]

        # For this example, assume query_kb returns a list of doc dicts:
        retrieved_docs = retrieved_indices

        st.subheader("Top roles recommanded")
        for d in retrieved_docs:
            st.write(f"**{d['title']}** — {d['desc']}")

        st.subheader("personalized guidance according to your skills")
        # Optional: a short resume summary to include
        resume_summary = (resume_data.get("Experience", "")[:400] + "...") if resume_data.get("Experience") else ""
        guidance = generate_guidance(skills_list, resume_summary, retrieved_docs)

        st.markdown("### Recommendation")
        st.write(guidance)
    else:
        st.warning("No skills found in resume. Please update your resume for better recommendations.")
