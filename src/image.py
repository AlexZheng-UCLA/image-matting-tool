import numpy as np
import os
import cv2
import copy
from PIL import Image, ImageDraw
import pyheif

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
            img_path = os.path.join(img_dir_path, filename)
            # open filename image as numpy
            img = Image.open(img_path)
            img_list.append(img)
            filename = filename.split(".")[0]
            filename_list.append(filename)

        elif filename.endswith(".heic"):
            heif_image = pyheif.read(filename)
            img = Image.frombytes(
                heif_image.mode,
                heif_image.size,
                heif_image.data,
                "raw",
                heif_image.mode,
                heif_image.stride,
            )
            img_list.append(img)
            filename = filename.split(".")[0]
            filename_list.append(filename)
    return img_list, filename_list, None

def convert_img_version(img, version="rgb"):
    img_copy = copy.deepcopy(img)
    if version=="rgb":
        img_copy = img_copy.convert("RGB")
    elif version=="rgba":
        img_copy = img_copy.convert("RGBA")

    return img_copy


def convert_img_np_version(img_np, version="rgb"):
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

def overlay_images(overlay_image, background_image, mask, real_width, ratio):

    width_bg, height_bg = background_image.size
    w_ratio, h_ratio = ratio[0], ratio[1]

    img_bg_copy = copy.deepcopy(background_image)
    img_bg_copy = convert_img_version(img_bg_copy, version="rgba")
    img_overlay_copy = copy.deepcopy(overlay_image)

    width_img, height_img = overlay_image.size
    scale = height_bg * h_ratio / height_img
    new_height = int(height_img * scale)
    new_width = int(width_img * scale)
    new_real_width = int(real_width * scale)

    img_resized = img_overlay_copy.resize((new_width, new_height))
    position = (int(width_bg * w_ratio - new_real_width/2), int(height_bg * (1-h_ratio)))
    img_pasted = Image.new('RGBA', img_bg_copy.size)

    img_pasted.paste(img_bg_copy, (0, 0))
    img_pasted.paste(img_resized, position, img_resized)  # Use img_resized as the mask
    return img_pasted


def paste_to_background(
    images,
    mask,
    background_list,
    ratios,
):  
    mask_vpos, mask_hpos = np.where(mask)
    mask_width = np.max(mask_hpos) - np.min(mask_hpos)
    mask_height = np.max(mask_vpos) - np.min(mask_vpos)
    # mask_median_hpos = np.median(mask_hpos).astype(int)
    # mask_median_vpos = np.median(mask_vpos).astype(int)

    img_paste_list = []
    for img_bg, ratio in zip(background_list, ratios):
        img_paste = overlay_images(images, img_bg, mask, mask_width, ratio)
        img_paste_list.append(img_paste)

    return img_paste_list


def move_masked_add_background(
    file_name, 
    save_dir, 
    img_np,
    background_list, 
    merged_masks,
    mask_option,
    h_shift, 
    save_image=True
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
            background_np = convert_img_np_version(background_np, version="rgba")
        elif background_np.shape[-1] > img_np_copy.shape[-1]:
            img_np_copy = convert_img_np_version(img_np_copy, version="rgba")

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

def background_add_box(image, filename, hc_ratio, vc_ratio):
    # Open the image file
    draw = ImageDraw.Draw(image)

    # Get image dimensions
    img_width, img_height = image.size

    # Calculate box dimensions and position
    box_width = 0.4 * img_width
    box_height = img_height * vc_ratio

    # Calculate box horizontal center
    box_hcenter = hc_ratio * img_width

    # Calculate the box left edge, top edge and right edge
    box_left = box_hcenter - (box_width / 2)
    box_right = box_hcenter + (box_width / 2)
    box_top = (1-vc_ratio) * img_height

    # Draw the box
    draw.rectangle(((box_left, box_top), (box_right, img_height)), outline='red', width=10)

    # Save the image with the box
    if not os.path.exists(f'/root/image-matting-tool/outputs/background_box'):
        os.mkdir(f'/root/image-matting-tool/outputs/background_box')
    image.save(f'/root/image-matting-tool/outputs/background_box/{filename}_box.jpg', 'JPEG')

def add_box_to_background(background_list, filename_list, ratios):
    for i, img_bg in enumerate(background_list):
        h_ratio, v_ratio = ratios[i][0], ratios[i][1]
        img_bg = background_add_box(img_bg, filename_list[i], h_ratio, v_ratio)



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
    background_dir = "/root/autodl-tmp/dataset/resorts-half"
    background_list, filename_list, _, = load_img_from_path(background_dir)

    ratios = [[0.8, 0.5],
              [0.75, 0.65],
              [0.75, 0.65]]
    add_box_to_background(background_list, filename_list, ratios)
    # save ratios to a json file
    import json
    with open(os.path.join(background_dir, "ratios.json"), 'w') as f:
        json.dump(ratios, f)

    



    
    
    
    