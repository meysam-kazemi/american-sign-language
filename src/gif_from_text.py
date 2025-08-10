



base_dir = 'dataset/asl_dataset'
sign_images = {}

for sign_name in os.listdir(base_dir):
    sign_dir = os.path.join(base_dir, sign_name)
    if os.path.isdir(sign_dir):
        image_files = [os.path.join(sign_dir, f) for f in os.listdir(sign_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        image_files.sort()  # Ensure images are in order
        sign_images[sign_name] = image_files
sign_images['SPACE'] = [base_dir+"/SPACE/space.jpeg"]

print(f"Loaded {len(sign_images)} signs.")
print(sign_images)


import string

def map_text_to_signs(text, sign_images):
    """
    Maps words in input text to available ASL signs.

    Args:
        text: The input text string.
        sign_images: A dictionary mapping sign names to image file paths.

    Returns:
        A list of mapped sign names.
    """
    # Convert to lowercase and remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    processed_text = text.lower().translate(translator)

    mapped_signs = []
    for letter in processed_text:
        if letter in sign_images:
            mapped_signs.append(letter)
        elif letter == ' ':
            mapped_signs.append('SPACE')

        # Option to handle missing words (currently skipping)
        # else:
        #     mapped_signs.append("FINGER_SPELL") # Placeholder for finger-spelling
    mapped_signs.append('SPACE')

    return mapped_signs

# Example usage (optional, for testing)
input_text = "Hello World 123"
mapped = map_text_to_signs(input_text, sign_images)
print(f"Input text: '{input_text}'")
print(f"Mapped signs: {mapped}")



def get_image_sequence(mapped_signs, sign_images):
    """
    Retrieves the sequence of image paths for a list of mapped signs.

    Args:
        mapped_signs: A list of mapped sign names.
        sign_images: A dictionary mapping sign names to image file paths.

    Returns:
        A list of image paths representing the entire sequence of signs.
    """
    image_sequence = []
    for sign_name in mapped_signs:
        if sign_name in sign_images:
            image_sequence.append(sign_images[sign_name][0])
        else:
            # This case should ideally not be hit with the current mapping,
            # but including for robustness. Could add a placeholder image here.
            print(f"Warning: Sign '{sign_name}' not found in dataset.")
            pass # Skip the sign if not found

    return image_sequence

# Example usage (optional, for testing)
mapped_signs_example = ['h', 'e', 'l', 'SPACE', 'l', 'o'] # Assuming these signs exist
sequence = get_image_sequence(mapped_signs_example, sign_images)
print(f"Generated image sequence with {len(sequence)} images.")
print(sequence[:10]) # Print first 10 paths as a sample



from PIL import Image

def create_animated_gif(image_paths, output_filename, duration=200, loop=0):
    """
    Creates an animated GIF from a sequence of images.

    Args:
        image_paths: A list of file paths to the images.
        output_filename: The name of the output GIF file.
        duration: The duration (in milliseconds) for each frame.
        loop: The number of times the GIF should loop (0 for infinite).
    """
    if not image_paths:
        print("No images provided to create GIF.")
        return

    # Open the first image
    first_image = Image.open(image_paths[0])

    # Create a list of subsequent images
    subsequent_images = []
    for path in image_paths[1:]:
        try:
            img = Image.open(path)
            subsequent_images.append(img)
        except FileNotFoundError:
            print(f"Warning: Image file not found: {path}. Skipping.")
        except Exception as e:
            print(f"Warning: Could not open image file: {path}. Error: {e}. Skipping.")


    # Save the first image as a GIF, appending subsequent images
    if subsequent_images:
        first_image.save(output_filename, save_all=True, append_images=subsequent_images, duration=duration, loop=loop)
    else:
         first_image.save(output_filename, save_all=True, duration=duration, loop=loop)


    print(f"Animated GIF saved as {output_filename}")

# Example usage (optional, for testing)
# Assuming 'sequence' variable holds a list of image paths from the previous step
if 'sequence' in locals() and sequence:
    create_animated_gif(sequence, "asl_animation.gif")
else:
    print("No image sequence available to create GIF. Run previous steps.")




# Example usage: Create a simple GIF for 'h', 'e', 'l', 'l', 'o'
input_text_example = "hello how are you my friend"
mapped_signs_example = map_text_to_signs(input_text_example, sign_images)
sequence_example = get_image_sequence(mapped_signs_example, sign_images)

if sequence_example:
    create_animated_gif(sequence_example, "asl_animation.gif", duration=300) # Adjust duration as needed
else:
    print("Could not generate image sequence for 'hello'. Cannot create GIF.")
