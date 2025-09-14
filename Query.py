import numpy as np 
import faiss
from sentence_transformers import SentenceTransformer
from build_kb import load_knowledge_base, build_to_index

#query function 
def query_kb(query , model , index, kb, top_k = 3, return_docs = False):
    query_vec = model.encode([query], convert_to_numpy = True)
    D, I = index.search(query_vec, top_k)

    results = []
    for idx in I[0]:
        if return_docs:
         results.append(kb[idx])

        else:
         results.append(kb[idx]["title"])

    return results