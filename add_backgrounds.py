import os
from PIL import Image, ImageEnhance, ImageFilter
import random

def add_background_and_lighting(input_folder, background_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    backgrounds = [f for f in os.listdir(background_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.webp'):
            input_path = os.path.join(input_folder, filename)
            
            # Open the foreground image
            with Image.open(input_path) as foreground:
                # Convert to RGBA if it's not already
                foreground = foreground.convert("RGBA")
                
                # Process with each background
                for bg_filename in backgrounds:
                    bg_path = os.path.join(background_folder, bg_filename)
                    
                    with Image.open(bg_path) as background:
                        # Convert background to RGBA and resize to match foreground
                        background = background.convert("RGBA").resize(foreground.size, Image.LANCZOS)
                        
                        # Create a new image with an alpha layer
                        composite = Image.new("RGBA", foreground.size, (0, 0, 0, 0))
                        
                        # Paste background onto the new image
                        composite.paste(background, (0, 0))
                        
                        # Paste foreground onto the new image, using its alpha channel as mask
                        composite.paste(foreground, (0, 0), foreground)
                        
                        # Apply random lighting effect
                        enhancer = ImageEnhance.Brightness(composite)
                        factor = 0.8 + random.random() * 0.4  # Random factor between 0.8 and 1.2
                        composite = enhancer.enhance(factor)
                        
                        # Apply subtle glow effect
                        glow = composite.filter(ImageFilter.GaussianBlur(radius=10))
                        composite = Image.blend(composite, glow, 0.2)
                        
                        # Save the result
                        output_filename = f"final_{os.path.splitext(filename)[0]}_{os.path.splitext(bg_filename)[0]}.webp"
                        output_path = os.path.join(output_folder, output_filename)
                        composite.save(output_path, "WEBP", quality=95)
                        print(f"Processed: {filename} with background {bg_filename}")

    print("Background and lighting effects added. Check the output folder for results.")

# Example usage

input_folder = './outputs'  # Folder with background-removed images
background_folder = './backgrounds'
output_folder = './final_outputs'
add_background_and_lighting(input_folder, background_folder, output_folder)