import json
from parse_resume import extract_text_from_pdf
from skills_extracter import skill_extracter

def structured_resume(file):
    text = extract_text_from_pdf(file)

    # Split sections (basic approach)
    sections = {}
    current = None
    lines = text.split("\n")

    for line in lines:
        line_clean = line.strip().lower()
        if "education" in line_clean:
            current = "Education"
            sections[current] = []
        elif "experience" in line_clean:
            current = "Experience"
            sections[current] = []
        elif "projects" in line_clean:
            current = "Projects"
            sections[current] = []
        elif "skills" in line_clean:
            current = "Skills"
            sections[current] = []
        elif current:
            sections[current].append(line.strip())

    for sec in sections:
        sections[sec] = " ".join(sections[sec])
    
    # Add extracted skills
    sections["Extracted_Skills"] = skill_extracter(text)

    return sections



