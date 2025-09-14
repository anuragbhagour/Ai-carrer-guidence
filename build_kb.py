import json
import faiss
import numpy as np 
from sentence_transformers import SentenceTransformer

#loading knowlegde base 
def load_knowledge_base(path = "knowledge_base.json"):
    with open(path, "r") as f:
        kb = json.load(f)
    return kb

#build embeddings
def build_to_index(kb, model_name = "all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)

    texts = [entry["desc"] for entry in kb]
    embeddings = model.encode(texts, convert_to_numpy = True)

    #build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return model, index, embeddings