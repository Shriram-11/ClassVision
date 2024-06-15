import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img


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


def augment_images(input_folder, output_folder, target_size, batch_size=32, save_prefix='augmented'):
    os.makedirs(output_folder, exist_ok=True)
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    image_files = [f for f in os.listdir(
        input_folder) if os.path.isfile(os.path.join(input_folder, f))]

    for image_file in image_files:
        img = load_img(os.path.join(input_folder, image_file))
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)

        i = 0
        for batch in datagen.flow(x, batch_size=batch_size, save_to_dir=output_folder, save_prefix=save_prefix, save_format='jpg'):
            i += 1
            if i >= 5:
                break


if __name__ == "__main__":
    categories = ["lecture", "no_lecture"]
    target_size = (224, 224)

    for category in categories:
        input_folder = os.path.join(category)
        resized_output_folder = os.path.join(f"{category}_resized")
        augmented_output_folder = os.path.join(f"{category}_augmented")

        resize_images(input_folder, resized_output_folder, target_size)
        augment_images(resized_output_folder,
                       augmented_output_folder, target_size)
