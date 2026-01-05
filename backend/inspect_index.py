import faiss
import os

# Path to the FAISS index file
INDEX_PATH = os.path.join(os.path.dirname(__file__), 'artifact_vectors.index')

def inspect_faiss_index():
    """
    Loads and inspects the FAISS index file.
    """
    if not os.path.exists(INDEX_PATH):
        print(f"Index file not found at: {INDEX_PATH}")
        return

    try:
        # Load the index from disk
        index = faiss.read_index(INDEX_PATH)

        print("--- FAISS Index Inspection ---")
        print(f"Index file: {os.path.basename(INDEX_PATH)}")
        
        # Get the number of vectors in the index
        num_vectors = index.ntotal
        print(f"Number of vectors stored: {num_vectors}")

        # Get the dimensionality of the vectors
        vector_dim = index.d
        print(f"Vector dimension: {vector_dim}")

        if num_vectors > 0:
            # You can't see the original text, but you can retrieve the vectors by their ID.
            # For example, to see the vector for the artifact with ID 1:
            try:
                # Note: The reconstruct method may not be available for all index types.
                # It works for IndexFlatL2 and IndexIDMap.
                vector = index.reconstruct(0) # Reconstruct the first vector (ID 0)
                print(f"Sample vector (first 10 dimensions) for ID 0: {vector[:10]}")
            except RuntimeError as e:
                print(f"Could not reconstruct vector. This index type may not support it. Error: {e}")
                print("The index still works for searching.")


    except Exception as e:
        print(f"An error occurred while reading the index: {e}")
        print("The file might be corrupted or not a valid FAISS index.")

if __name__ == '__main__':
    inspect_faiss_index()
