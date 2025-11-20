import streamlit as st
from structured_resume import structured_resume
from build_kb import load_knowledge_base, build_to_index
from Query import query_kb
from rag_pipeline import generate_guidance

# CACHE heavy resources so they survive Streamlit reruns
@st.cache_resource
def init_kb():
    kb = load_knowledge_base()
    model, index, embeddings = build_to_index(kb)
    return kb, model, index, embeddings

kb, model, index, embeddings = init_kb()

st.title("AI Career Guidance Bot")

uploaded_file = st.file_uploader("Upload your Resume (PDF)", type="pdf")

# Persist resume parsing in session_state so chat can use it later
if uploaded_file is not None:
    resume_data = structured_resume(uploaded_file)
    st.session_state['resume_data'] = resume_data
    st.session_state['skills_list'] = resume_data.get("Extracted_Skills", [])
    st.session_state['skills_str'] = ", ".join(st.session_state['skills_list'])

# Safe defaults if user hasn't uploaded yet
resume_data = st.session_state.get('resume_data', {})
skills_list = st.session_state.get('skills_list', [])
skills_str = st.session_state.get('skills_str', "")

if skills_list:
    top_k = 3
    # Use a clearer variable name
    retrieved_docs = query_kb(skills_str, model, index, kb, top_k=top_k, return_docs=True)

    st.subheader("Top roles recommended")
    for d in retrieved_docs:
        st.write(f"**{d.get('title','-')}** — {d.get('desc','-')}")

    st.subheader("Personalized guidance according to your skills")
    resume_summary = (resume_data.get("Experience", "")[:400] + "...") if resume_data.get("Experience") else ""
    try:
        with st.spinner("Generating guidance..."):
            guidance = generate_guidance(skills_list, resume_summary, retrieved_docs)
        st.markdown("### Recommendation")
        st.write(guidance)
    except Exception as e:
        st.error("Couldn't generate guidance right now. Try again later.")
        st.error(str(e))
else:
    st.warning("No skills found in resume. Please upload or update your resume for better recommendations.")

# Chat area (session-safe)
st.subheader("Chat with Career Bot")
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask me about your career path..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Use safe defaults for retrieval
    retrieved_docs = query_kb(skills_str, model, index, kb, top_k=3, return_docs=True) if skills_str else []

    try:
        with st.spinner("Thinking..."):
            guidance = generate_guidance(
                skills_list,
                resume_data.get("Experience", ""),
                retrieved_docs,
                conversation=st.session_state.messages,
                user_goal=prompt
            )
    except Exception as e:
        guidance = "Sorry, I couldn't generate a response at the moment."

    with st.chat_message("assistant"):
        st.write(guidance)
    st.session_state.messages.append({"role": "assistant", "content": guidance})
