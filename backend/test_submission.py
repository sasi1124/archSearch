import requests
import os

# --- Test Configuration ---
BASE_URL = "http://127.0.0.1:5000"
# The script is in 'backend', so the path is relative to it.
IMAGE_PATH = os.path.join(os.path.dirname(__file__), 'data', 'rosetta.jpeg')

def test_submit_artifact():
    """
    Tests the /api/artifacts endpoint by sending a sample artifact submission.
    """
    print("--- Running Artifact Submission Test ---")
    
    # 1. Check if the test image exists
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ ERROR: Test image not found at {IMAGE_PATH}")
        print("Please ensure the file 'rosetta.jpeg' is in the 'data' directory.")
        return

    # 2. Define the form data for the artifact
    artifact_data = {
        'name': 'Test Artifact - Rosetta Stone',
        'description': 'A slab of granodiorite with a decree from 196 BC.',
        'category': 'Epigraphy',
        'location': 'British Museum, London',
        'uploadedBy': 'testuser',
        'license': 'CC-BY-SA',
        'date': '196 BC'
    }
    
    # 3. Define the files to be uploaded
    files = {
        'images': ('rosetta.jpeg', open(IMAGE_PATH, 'rb'), 'image/jpeg')
    }
    
    # 4. Send the POST request
    url = f"{BASE_URL}/api/artifacts"
    print(f"Sending POST request to {url} with data and image...")
    
    try:
        response = requests.post(url, data=artifact_data, files=files)
        
        # 5. Analyze the response
        print(f"Received status code: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ SUCCESS: Artifact submitted successfully.")
            print("Response JSON:", response.json())
        elif response.status_code == 500:
            print("❌ FAILURE: Received 500 Internal Server Error.")
            print("This indicates a crash in the backend `submit_artifact` function.")
            # The response text for a 500 error is usually an HTML page with a traceback.
            # This traceback is the key to debugging the problem.
            print("\n--- Server Error Traceback ---\n")
            print(response.text)
            print("\n--- End of Traceback ---")
        else:
            print(f"Received unexpected status code: {response.status_code}")
            print("Response content:", response.text)
            
    except requests.exceptions.ConnectionError as e:
        print(f"❌ CONNECTION ERROR: Could not connect to the server at {BASE_URL}.")
        print("Please ensure the Flask backend server (`app.py`) is running.")
    except Exception as e:
        print(f"An unexpected error occurred during the test: {e}")

if __name__ == '__main__':
    test_submit_artifact()
