import numpy as np
import os
import cv2
import copy
from PIL import Image

def load_img_from_path(img_dir_path):
    if img_dir_path is None:
        return False, False, "Please provide path to image directory"
    if not os.path.exists(img_dir_path):
        return False, False, "image directory dosen't exist!"

    img_list = []
    filename_list = []
    # append all the img in the img_path to the list
    for filename in os.listdir(img_dir_path):
        desired_ext = [".jpg", ".png", ".jpeg",  ".JPG", ".PNG", ".JPEG"]
        if any(filename.endswith(ext) for ext in desired_ext):
            filename_list.append(filename)
            img_path = os.path.join(img_dir_path, filename)
            # open filename image as numpy
            img = Image.open(img_path)
            img_list.append(img)
    return img_list, filename_list, None


def unify_image_version(img_np, version="rgb"):
    img_np_copy = copy.deepcopy(img_np)
    if version=="rgb":
        if len(img_np_copy.shape) == 2:  # Grayscale image
            # Convert to RGB
            img_np_copy = np.stack((img_np_copy,) * 3, axis=-1)
        elif len(img_np_copy.shape) == 3 and img_np_copy.shape[-1] == 4:  # RGBA image
            # Convert to RGB by discarding the alpha channel
            img_np_copy = img_np_copy[:, :, :3]
        return img_np_copy

    elif version=="rgba":
        if len(img_np_copy.shape) == 2:  # Grayscale image
            # Convert to RGBA by repeating grayscale for R, G, B, and setting A to max value
            img_np_copy = np.stack((img_np_copy,) * 3 + (np.full(img_np_copy.shape, 255),), axis=-1)
        elif len(img_np_copy.shape) == 3 and img_np_copy.shape[-1] == 3:  # RGB image
            # Convert to RGBA by adding a new alpha channel with max value
            alpha_channel = np.full((img_np_copy.shape[0], img_np_copy.shape[1]), 255)
            img_np_copy = np.dstack((img_np_copy, alpha_channel))

    return img_np_copy


def move_masked_add_background(
    file_name, 
    save_dir, 
    img_np,
    background_list, 
    merged_masks,
    mask_option,
    h_shift, 
    save_image=True,
):
    msg = ""
    width, height = img_np.shape[1], img_np.shape[0]
    size = max(width, height)
    if not merged_masks:
        raise ValueError("merged_masks is empty")
    try:
        if mask_option == "first":
            mask = merged_masks[0]
        elif mask_option in ["1", "2", "3"]:
            mask = merged_masks[int(mask_option)-1]
        elif mask_option == "largest":
            mask = merged_masks[np.argmax([np.sum(mask) for mask in merged_masks])]
        elif mask_option == "smallest":
            mask = merged_masks[np.argmin([np.sum(mask) for mask in merged_masks])]
        elif mask_option == "merge":
            mask = np.sum(merged_masks, axis=0)
        else:
            mask = merged_masks[0]
            msg += "mask_option is not valid, use the first mask as default"
    except IndexError:
        raise ValueError(f"merged_masks does not have enough elements for mask_option {mask_option}")


    processed_images = []
    img_np_copy = copy.deepcopy(img_np)

    mask_vpos, mask_hpos = np.where(mask)
    mask_width = np.max(mask_hpos) - np.min(mask_hpos)
    mask_height = np.max(mask_vpos) - np.min(mask_vpos)
    mask_median_hpos = np.median(mask_hpos).astype(int)
    mask_median_vpos = np.median(mask_vpos).astype(int)

    if h_shift:
        if width // 2 - mask_median_hpos > 0: # right shft
            hshift = min(width // 2 - mask_median_hpos, width - mask_width)
        else:                                 # left shift 
            hshift = max(width // 2 - mask_median_hpos, mask_width - width)

        msg += f"mask_width: {mask_width}, mask_height: {mask_height}, mask_median_hpos: {mask_median_hpos}, mask_median_vpos: {mask_median_vpos}, hshift: {hshift}"
    else:
        hshift = 0

    for idx, background in enumerate(background_list):
        
        background = background.resize((size, size))
        background_np = np.array(background)
        background_np = background_np[:height, :width]

        if background_np.shape[-1] < img_np_copy.shape[-1]:
            background_np = unify_image_version(background_np, version="rgba")
        elif background_np.shape[-1] > img_np_copy.shape[-1]:
            img_np_copy = unify_image_version(img_np_copy, version="rgba")

        for i in range(width):
            for j in range(height):
                if mask[i, j] and 0 <= j+hshift < width:
                    background_np[i, j+hshift, :] = img_np_copy[i, j, :]

        background_np = background_np.astype(np.uint8)
        img_processed = Image.fromarray(background_np)
        if save_image:
            img_processed.save(os.path.join(save_dir, f"{file_name}_processed_{idx}.png"))

        processed_images.append(img_processed)
    return processed_images, msg


def load_background_from_path(background_dir):
    if background_dir is None:
        return None, "Please provide path to background directory"
    if not os.path.exists(background_dir):
        return None, "background directory dosen't exist!"
    
    background_list = []
    # append all the img in the img_path to the list
    for filename in os.listdir(background_dir):
        desired_ext = [".jpg", ".png", ".jpeg",  ".JPG", ".PNG", ".JPEG"]
        if any(filename.endswith(ext) for ext in desired_ext):
            background_path = os.path.join(background_dir, filename)
            # open filename image as numpy
            background = Image.open(background_path)
            background_list.append(background)
    return background_list, "Load background from path successfully"


def generate_pure_background(width, height, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    colors = {
        'black': [0, 0, 0],
        'white': [255, 255, 255],
        'red': [255, 0, 0],
        'green': [0, 255, 0],
        'blue': [0, 0, 255],
    }

    # Generate images for each color
    for color, rgb in colors.items():
        array = np.full((1024, 1024, 3), rgb, dtype=np.uint8)
        img = Image.fromarray(array)
        img.save(os.path.join(save_dir, f'{color}_image.png'))


if __name__ == "__main__":
    # generate_pure_background(1024, 1024, save_dir="/root/fstudio/backgrounds")
    background_list, msg = load_background_from_path("/root/fstudio/backgrounds")
    if background_list is None:
        print(msg)
    
    
    
    