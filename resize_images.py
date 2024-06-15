import os
from PIL import Image


def resize_images(input_folder, output_folder, target_size):
    os.makedirs(output_folder, exist_ok=True)
    image_files = [f for f in os.listdir(
        input_folder) if os.path.isfile(os.path.join(input_folder, f))]

    for image_file in image_files:
        try:
            with Image.open(os.path.join(input_folder, image_file)) as img:
                resized_img = img.resize(target_size, Image.LANCZOS)
                resized_img.save(os.path.join(output_folder, image_file))
        except Exception as e:
            print(f"Error processing {image_file}: {e}")


if __name__ == "__main__":
    # Example usage:
    categories = ["lecture", "no_lecture"]
    target_size = (224, 224)

    for category in categories:
        input_folder = os.path.join(category)
        resized_output_folder = os.path.join(f"{category}_resized")

        resize_images(input_folder, resized_output_folder, target_size)
