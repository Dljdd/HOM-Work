import os
from PIL import Image, ImageEnhance, ImageFilter

def add_background_and_lighting(input_folder, background_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    backgrounds = [f for f in os.listdir(background_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            foreground = Image.open(input_path).convert("RGBA")

            # Randomly choose a background
            bg_filename = backgrounds[hash(filename) % len(backgrounds)]
            bg_path = os.path.join(background_folder, bg_filename)
            background = Image.open(bg_path).convert("RGBA")

            # Resize background to match foreground
            background = background.resize(foreground.size, Image.LANCZOS)

            # Composite the images
            composite = Image.alpha_composite(background, foreground)

            # Apply random lighting effect
            enhancer = ImageEnhance.Brightness(composite)
            factor = 0.8 + (hash(filename) % 5) / 10  # Random factor between 0.8 and 1.2
            composite = enhancer.enhance(factor)

            # Apply subtle glow effect
            glow = composite.filter(ImageFilter.GaussianBlur(radius=10))
            composite = Image.blend(composite, glow, 0.2)

            # Save the result
            output_path = os.path.join(output_folder, f"final_{filename}")
            composite = composite.convert("RGB")  # Convert to RGB for saving as JPEG
            composite.save(output_path, "JPEG", quality=95)
            print(f"Processed: {filename}")

    print("Background and lighting effects added. Check the output folder for results.")

# Usage
input_folder = 'outputs'  # Folder with background-removed images
background_folder = '/path/to/background/images'
output_folder = 'final_outputs'
add_background_and_lighting(input_folder, background_folder, output_folder)