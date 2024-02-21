import os
from glob import glob
from PIL import Image

def rescale_image_and_mask(input_image_path, output_image_path, scale_factor):
    # Load the image
    image = Image.open(input_image_path)
    
    # Resize the image
    new_size = (image.width // scale_factor, image.height // scale_factor)
    resized_image = image.resize(new_size, Image.ANTIALIAS)
    
    # Save the resized image
    resized_image.save(output_image_path)

if __name__ == "__main__":
    # Set the paths
    image_folder = "/Volumes/dataset/paip/"
    mask_folder = "/Volumes/dataset/paip/mask/"
    output_folder = "/Volumes/dataset/paip/data/rescaled/"

    # Define the scale factors
    scale_factors = [1, 2, 8, 16]

    # Glob for image files
    image_files = glob(os.path.join(image_folder, "*.svs"))

    # Iterate through each image file
    for image_file in image_files:
        # Get the corresponding mask file
        mask_file = os.path.join(mask_folder, os.path.basename(image_file).replace(".svs", ".tif"))

        # Iterate through each scale factor
        for scale_factor in scale_factors:
            # Create a subfolder for each scale factor
            scale_folder = os.path.join(output_folder, f"scale_{scale_factor}/")
            os.makedirs(scale_folder, exist_ok=True)

            # Rescale and save the image
            output_image_path = os.path.join(scale_folder, os.path.basename(image_file))
            rescale_image_and_mask(image_file, output_image_path, scale_factor)

            # Rescale and save the mask
            output_mask_path = os.path.join(scale_folder, os.path.basename(mask_file))
            rescale_image_and_mask(mask_file, output_mask_path, scale_factor)