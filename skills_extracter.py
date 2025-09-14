import spacy
from fuzzywuzzy import fuzz
from skill_list import skills

nlp = spacy.load("en_core_web_sm")

def skill_extracter(text):
    doc = nlp(text)
    extracted = set()

    for token in doc:
        for skill in skills:
            if fuzz.ratio(token.text.lower() , skill.lower()) > 80:
                extracted.add(skill)
    return list(extracted)