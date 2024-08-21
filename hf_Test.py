import requests
from PIL import Image
from io import BytesIO
import os
from datetime import datetime

API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"
headers = {"Authorization": "Bearer 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"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
    return response.content

def save_image(image_bytes, folder="generated_images"):
    if not image_bytes:
        raise ValueError("No image data received")

    # Create the folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Generate a unique filename using timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"generated_image_{timestamp}.png"
    filepath = os.path.join(folder, filename)
    
    # Save the image
    try:
        image = Image.open(BytesIO(image_bytes))
        image.save(filepath)
        print(f"Image saved as {filepath}")
        return filepath
    except Exception as e:
        print(f"Failed to save image: {e}")
        print(f"Response content type: {type(image_bytes)}")
        print(f"Response content (first 100 bytes): {image_bytes[:100]}")
        raise

try:
    # Generate the image
    print("Sending request to API...")
    image_bytes = query({
        "inputs": "Indian woman vintage photography of a young woman as a ((fashion model)) with ((long hair)),dressed in white sports bra and cycling shorts,kneeling position,hand in hair,confident gaze,slight smile,studio,studio lighting with strong rim light,(hero view:1.1)",
    })
    
    print(f"Received response. Content length: {len(image_bytes)} bytes")

    # Save the image
    saved_filepath = save_image(image_bytes)

    # Display the image
    image = Image.open(saved_filepath)
    image.show()

except Exception as e:
    print(f"An error occurred: {e}")