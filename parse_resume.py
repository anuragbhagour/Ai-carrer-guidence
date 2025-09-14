import pdfplumber

def extract_text_from_pdf(file):
    doc = pdfplumber.open(file)
    text = ""
    for page in doc.pages:
        text += page.extract_text() + "\n"
    doc.close()
    return text

