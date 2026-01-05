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

        # If this is an IDMap, we can print the number of stored IDs
        try:
            if hasattr(index, 'id_map'):
                # Some wrappers expose id_map
                pass
        except Exception:
            pass

        if num_vectors > 0:
            # Many index types support reconstruct for flat indexes.
            try:
                vector = index.reconstruct(0)
                print(f"Sample vector (first 10 dims) for internal row 0: {vector[:10]}")
            except Exception as e:
                print(f"Could not reconstruct vector. Error: {e}")


    except Exception as e:
        print(f"An error occurred while reading the index: {e}")
        print("The file might be corrupted or not a valid FAISS index.")

if __name__ == '__main__':
    inspect_faiss_index()
