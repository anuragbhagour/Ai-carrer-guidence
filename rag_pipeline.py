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





def build_rag_prompt(skills, resume_summary, retrieved_docs, conversation=None, user_goal=None):
    """
    Build the prompt for Gemini with resume context, KB, and chat history.
    """
    skills_line = ", ".join(skills) if skills else "No explicit skills extracted."
    prompt_parts = [
        "You are an expert career guidance assistant. Always give practical, actionable advice.",
        "",
        f"Candidate skills: {skills_line}",
        f"Resume summary: {resume_summary or 'N/A'}",
        "",
        "Relevant career knowledge base (use only as factual context):"
    ]
    for i, doc in enumerate(retrieved_docs, 1):
        prompt_parts.append(f"{i}. {doc['title']} — {doc['desc']}")

    # Add conversation history
    if conversation:
        prompt_parts.append("\nConversation so far:")
        for turn in conversation:
            role = turn["role"].capitalize()
            content = turn["content"]
            prompt_parts.append(f"{role}: {content}")

    # Current user request
    if user_goal:
        prompt_parts.append(f"\nUser latest question/request: {user_goal}")

    prompt_parts.append(
        "\nNow answer as a helpful career counselor: recommend paths, missing skills, and learning steps. "
        "Keep answers clear and structured."
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


def generate_guidance(skills, resume_summary, retrieved_docs, conversation=None, user_goal=None):
    """
    Generate guidance using Gemini, with optional conversation history.
    """
    prompt = build_rag_prompt(skills, resume_summary, retrieved_docs, conversation=conversation, user_goal=user_goal)
    guidance = call_gemini(prompt)
    return guidance.strip()
