from retriever import retrieve

# Then you can call retrieve
retrieved_docs, retrieved_doc_names, reranked_results = retrieve("What is a Gommory cut?")

# Print results
for i, doc_name in enumerate(retrieved_doc_names[:3]):
    score = reranked_results[i].relevance_score
    print(f"{i+1}. {doc_name} (Score: {score:.2f})")