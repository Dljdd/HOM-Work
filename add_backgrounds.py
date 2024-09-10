import os
from PIL import Image, ImageEnhance, ImageFilter

def add_background_and_lighting(input_folder, background_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    backgrounds = [f for f in os.listdir(background_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.png'):
            input_path = os.path.join(input_folder, filename)
            foreground = Image.open(input_path).convert("RGBA")

            # Randomly choose a background
            # bg_filename = backgrounds[hash(filename) % len(backgrounds)]
            index=1
            for bg_filename in os.listdir(background_folder):
                if bg_filename.lower().endswith('.webp'):
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

                    # Create a folder for each background
                    if not os.path.exists(f"{output_folder}/{index}"):
                        os.makedirs(f"{output_folder}/{index}")
                    # Save the result
                    output_path = os.path.join(f"{output_folder}/{index}", f"final_{filename}")
                    composite.save(output_path, "WEBP", quality=95)
                    index+=1
                    print(f"Processed: {filename}")

    print("Background and lighting effects added. Check the output folder for results.")

# Example usage
input_folder = './nobg'  # Folder with background-removed images
background_folder = './backgrounds'
output_folder = './final_outputs'
add_background_and_lighting(input_folder, background_folder, output_folder)