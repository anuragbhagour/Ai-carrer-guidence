import os 
from dotenv import load_dotenv
import google.generativeai as genai
from typing import List, Dict

#configure gemini api key from enviroment var
load_dotenv()
GEN_KEY = os.getenv("GEMINI_API_KEY")
if not GEN_KEY:
    raise EnvironmentError("set the gen ai key")
genai.configure(api_key = GEN_KEY)
GEMINI_MODEL = "gemini-2.5-pro"

def build_rag_prompt(skills:List[str] , resume_summary: str, retrieved_docs = List[Dict] , user_goal:str =None) -> str:
    skills_line = ", ".join(skills) if skills else "explicit skills extracted"

    prompt_parts = [
        "You are an expert career and learning-path advisor. Use the information below to recommend career paths and a practical learning roadmap.",
        "",
        f"Candidate skills: {skills_line}",
        f"Candidate short resume summary: {resume_summary or 'N/A'}",
    ]
    if user_goal:
        prompt_parts.append(f"user goals/prefrence : {user_goal}")
    
    prompt_parts.append("")
    prompt_parts.append("Retrieved knowledge (from the KB). Use these as factual context — DO NOT hallucinate other facts. Numbered list:")
    for i, doc in enumerate(retrieved_docs, 1):
        title = doc.get("title", f"doc_{i}")
        desc = doc.get("desc", "")
        prompt_parts.append(f"{i}. {title} — {desc}")

    prompt_parts.append("")
    prompt_parts.append(
        "Task: 1) Recommend the top 2-3 suitable career paths and explain WHY each is a good fit given the candidate's skills. "
        "2) For each recommended path, list the top 4 concrete skills/tools to learn next (ordered), and 3 practical project ideas or small steps to demonstrate those skills. "
        "3) Give a concise suggested learning roadmap (3-month and 6-month milestones). "
        "Keep the answer actionable, concise, and prioritize high-impact items. If the candidate needs prerequisites, mention them briefly."
    )
    return "\n".join(prompt_parts)


def call_gemini(prompt: str, max_output_tokens: int = 512, temperature: float = 0.0) -> str:
    """
    Call Gemini using google-generativeai. Returns the assistant text.
    """
    # Create a generation request. SDK surface can vary; this follows the genai usage pattern.
    model = genai.models.get(GEMINI_MODEL) if hasattr(genai, "models") else genai.GenerativeModel(GEMINI_MODEL)
    # Try two different call forms to be more robust to SDK variations
    try:
        # Preferred: model.generate() / generate_content depending on SDK
        resp = model.generate(prompt=prompt, temperature=temperature, max_output_tokens=max_output_tokens)
        # Extract text depending on response structure
        if isinstance(resp, dict) and "candidates" in resp:
            return resp["candidates"][0].get("content", "")
        # some SDKs return object with .text or .candidates
        return getattr(resp, "text", str(resp))
    except Exception:
        # Fallback call
        resp = model.generate_content(prompt)
        # resp may have .text or .candidates
        return getattr(resp, "text", None) or (resp.candidates[0].content if getattr(resp, "candidates", None) else str(resp))


def generate_guidance(skills: List[str], resume_summary: str, retrieved_docs: List[Dict], user_goal: str = None) -> str:
    """
    High-level function to produce the RAG-generated guidance.
    """
    prompt = build_rag_prompt(skills, resume_summary, retrieved_docs, user_goal=user_goal)
    guidance = call_gemini(prompt)
    return guidance.strip()
