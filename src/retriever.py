import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from init import vo
from embedder import document_names, documents, documents_embeddings

def k_nearest_neighbors(query_embedding, k=10):
    query_embedding = np.array(query_embedding)
    documents_embeddings = np.array(documents_embeddings)
    query_embedding = query_embedding.reshape(1, -1)
    cosine_sim = cosine_similarity(query_embedding, documents_embeddings)
    sorted_indices = np.argsort(cosine_sim[0])[::-1]
    top_k_related_indices = sorted_indices[:k]
    top_k_related_embeddings = documents_embeddings[sorted_indices[:k]]
    top_k_related_embeddings = [list(row[:]) for row in top_k_related_embeddings]
    return top_k_related_indices

def retrieve(query):
    query_embedding = vo.embed([query], model="voyage-3", input_type="query").embeddings[0]
    retrieved_embd_indices = k_nearest_neighbors(
        query_embedding)
    print(retrieved_embd_indices)
    retrieved_docs = [documents[index] for index in retrieved_embd_indices]
    retrieved_doc_names = [document_names[index] for index in retrieved_embd_indices]

    documents_reranked = vo.rerank(
        query,
        retrieved_docs,
        model="rerank-lite-1",
        top_k=4
    )

    reranked_results = sorted(
        documents_reranked.results,
        key=lambda x: x.relevance_score,
        reverse=True
    )
    return retrieved_docs, retrieved_doc_names, reranked_results
