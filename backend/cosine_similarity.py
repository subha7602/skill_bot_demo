
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def cosine_similarity(vector_a, vector_b):
    """
    Calculate cosine similarity between two vectors.
    
    This function is used when sklearn is not available.
    
    Args:
        vector_a: First vector or list of vectors
        vector_b: Second vector or list of vectors
        
    Returns:
        float or array: Similarity score(s) between 0 and 1
    """
    # Convert to numpy arrays if they aren't already
    import numpy as np
    a = np.array(vector_a)
    b = np.array(vector_b)
    
    # Ensure both are 2D arrays
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    
    # Normalize the vectors
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    
    # Avoid division by zero
    a_norm = np.where(a_norm == 0, 1e-10, a_norm)
    b_norm = np.where(b_norm == 0, 1e-10, b_norm)
    
    a_normalized = a / a_norm
    b_normalized = b / b_norm
    
    # Calculate similarity
    similarity = np.dot(a_normalized, b_normalized.T)
    
    # If the input was single vectors, return a float
    if similarity.shape == (1, 1):
        return float(similarity[0, 0])
    else:
        return similarity