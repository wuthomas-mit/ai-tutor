import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Optional
import math

def evaluate_retrieval(relevancy_scores: List[int]) -> Tuple[int, float, Optional[int], float, bool]:
    first_score = relevancy_scores[0] if relevancy_scores else 0
    top_4_scores = relevancy_scores[:4]
    relevant_files = [score for score in top_4_scores if score >= 3]
    precision_at_4 = len(relevant_files) / 4 if len(top_4_scores) == 4 else 0.0
    
    first_index_of_4 = None
    for i, score in enumerate(relevancy_scores):
        if score == 4:
            first_index_of_4 = i + 1
            break
    
    dcg_at_4 = 0.0
    for i in range(min(4, len(relevancy_scores))):
        score = relevancy_scores[i]
        dcg_at_4 += score / math.log2(i + 2)
    
    best_relevancy = max(relevancy_scores) if relevancy_scores else 0
    average_relevancy = sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else 0.0
    
    return first_score, precision_at_4, best_relevancy, average_relevancy, first_index_of_4, dcg_at_4

def battery_test_retrieval(relevancy):
    result = []
    for scores in relevancy:
        result.append(evaluate_retrieval(scores))
    return result

def plot_retrieval_results(results):
    results_array = np.array(results)
    normalized_scores = np.clip(results_array[:, [0, 2, 3]], 1, 4)
    precision_at_4 = results_array[:, 1].reshape(-1, 1)
    precision_at_4 = 1 + 3 * precision_at_4
    
    first_index_of_4 = 4 - (results_array[:, 4] - 1)
    first_index_of_4 = first_index_of_4.reshape(-1, 1)
    
    dcg_at_4 = results_array[:, 5]
    min_val = np.min(dcg_at_4)
    max_val = np.max(dcg_at_4)
    
    if max_val - min_val > 0:
        normalized_dcg_at_4 = 1 + 3 * (dcg_at_4 - min_val) / (max_val - min_val)
    else:
        normalized_dcg_at_4 = np.ones_like(dcg_at_4) * 2
    
    normalized_dcg_at_4 = normalized_dcg_at_4.reshape(-1, 1)
    
    full_normalized_matrix = np.hstack((
        normalized_scores[:, [0]],
        precision_at_4,
        normalized_scores[:, [1]],
        normalized_scores[:, [2]],
        first_index_of_4,
        normalized_dcg_at_4
    ))
    
    fig, ax = plt.subplots(figsize=(10, len(results) * 0.5))
    cax = ax.matshow(full_normalized_matrix, cmap='RdYlGn', aspect='auto', vmin=1, vmax=4)
    
    for i in range(full_normalized_matrix.shape[0]):
        for j in range(full_normalized_matrix.shape[1]):
            ax.text(j, i, f'{results_array[i, j]:.2f}', ha='center', va='center', 
                   color='black', fontsize=8)
    
    ax.set_xticks(np.arange(full_normalized_matrix.shape[1]))
    ax.set_yticks(np.arange(full_normalized_matrix.shape[0]))
    ax.set_xticklabels(['first_score', 'precision_at_4', 'best_relevancy', 
                        'average_relevancy', 'first_index_of_4', 'dcg_at_4'], 
                       rotation=45, ha='right')
    ax.set_yticklabels([f'Q{i+1}' for i in range(full_normalized_matrix.shape[0])])
    
    plt.tight_layout()
    plt.show()