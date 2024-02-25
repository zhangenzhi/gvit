import os
import glob
import openslide
import tifffile
from PIL import Image

def rescale_and_save(image_path, mask_path, output_directory, target_size, zoom_level):
    # OpenSlide for SVS images
    slide = openslide.open_slide(image_path)

    # Calculate the dimensions at the specified zoom level
    scaled_width = slide.level_dimensions[zoom_level][0]
    scaled_height = slide.level_dimensions[zoom_level][1]

    # Read the region at the specified zoom level
    scaled_image = slide.read_region((0, 0), zoom_level, (scaled_width, scaled_height))
    scaled_image = scaled_image.resize(target_size, Image.Resampling.LANCZOS)

    # Rescale the TIFF mask
    mask = tifffile.imread(mask_path)
    scaled_mask = Image.fromarray(mask).resize(target_size, Image.Resampling.LANCZOS)

    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Save the rescaled images
    output_image_path = os.path.join(output_directory, f"rescaled_image_{zoom_level}_{target_size[0]}x{target_size[1]}.png")
    output_mask_path = os.path.join(output_directory, f"rescaled_mask_{zoom_level}_{target_size[0]}x{target_size[1]}.png")

    scaled_image.save(output_image_path)
    scaled_mask.save(output_mask_path)

    # Close the OpenSlide image
    slide.close()

if __name__ == "__main__":
    # Input directory paths
    image_directory = "/Volumes/data/dataset/paip/"
    mask_directory = "/Volumes/data/dataset/paip/mask"
    output_directory = "/Volumes/data/dataset/paip/output_images_and_masks/"
    os.makedirs(output_directory, exist_ok=True)
    processed_slides = os.listdir(output_directory)
    # Target sizes for rescaling
    target_sizes = [(1024, 1024), (512, 512)]

    # Zoom level for rescaling
    zoom_level = 0  # Adjust the zoom level as needed

    # Create a list of image and mask pairs
    image_mask_pairs = []
    for image_path in glob.glob(os.path.join(image_directory, "*.svs")):
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        if base_name in processed_slides:
            continue
        mask_path = os.path.join(mask_directory, f"{base_name}_mask.tif")
        if os.path.exists(mask_path):
            image_mask_pairs.append((image_path, mask_path))

    # Iterate over each pair and rescale
    for idx, (image_path, mask_path) in enumerate(image_mask_pairs, 1):
        print(f"Processing pair {idx}/{len(image_mask_pairs)}:")
        print(f"Image: {image_path}")
        print(f"Mask: {mask_path}")

        # Output directory for each pair
        output_directory = f"/Volumes/data/dataset/paip/output_images_and_masks/{os.path.basename(image_path)[:-4]}"

        for size in target_sizes:
            print(f"Rescaling to {size[0]}x{size[1]} at zoom level {zoom_level}")
            rescale_and_save(image_path, mask_path, output_directory, size, zoom_level)

        print("Processing complete.\n")
