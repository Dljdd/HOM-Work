import os
from PIL import Image
from rembg import new_session, remove

def remove_background(input_folder):
    # Create the output folder if it doesn't exist
    output_folder = "outputs"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process all images in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"bg_removed_{filename}")

            # Open the image and remove the background
            input_image = Image.open(input_path)
            output_image = remove(input_image)

            # Save the output image
            output_image.save(output_path)
            print(f"Processed: {filename}")

    print("Background removal complete. Check the 'outputs' folder for results.")

input_folder = 'a_photo_of_WH1TE_HOM'
remove_background(input_folder)