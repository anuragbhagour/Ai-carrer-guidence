import numpy as np 


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