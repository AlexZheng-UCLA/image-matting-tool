import torch
import os
from PIL import Image
import numpy as np
import copy
import cv2
import shutil
from collections import OrderedDict
from scipy.ndimage import binary_dilation
from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
sam_model_cache = OrderedDict()
sam_model_dir = "/root/stable-diffusion-webui/extensions/sd-webui-segment-anything/models/sam"
sam_model = {
    "H" : {
        "name": "vit_h",
        "checkpoint": "sam_vit_h_4b8939.pth"
    },
    # "L": {
    #     "name": "vit_l",
    #     "checkpoint": "sam_vit_l_0b3195.pth"
    # },
    # "B": {
    #     "name": "vit_b",
    #     "checkpoint": "sam_vit_b_01ec64.pth"
    # },
}

def load_sam_model(model_type):
    if model_type is None:
        return False, "Please provide path to sam model!"
    if not model_type in sam_model:
        return False, "Provide correct sam model type"
    
    if model_type in sam_model_cache:
        sam = sam_model_cache[model_type]
        msg = f"load sam {model_type} from sam_model_cache"
    else:
        model_path = os.path.join(sam_model_dir, sam_model[model_type]["checkpoint"])
        sam = sam_model_registry[sam_model[model_type]["name"]](checkpoint=model_path)
        msg = f"load sam {model_type} from checkpoint"
        sam_model_cache[model_type] = sam

    sam.to(device=device)
    sam.eval()
    return sam, msg

def mask_entire_image(sam_model, image):
    predictor = SamPredictor(sam_model)
    mask_generator = SamAutomaticMaskGenerator(sam_model)
    masks = mask_generator.generate(image)
    return masks


def dilate_mask(mask, dilation_amt):
    x, y = np.meshgrid(np.arange(dilation_amt), np.arange(dilation_amt))
    center = dilation_amt // 2
    dilation_kernel = ((x - center)**2 + (y - center)**2 <= center**2).astype(np.uint8)
    dilated_binary_img = binary_dilation(mask, dilation_kernel)
    dilated_mask = Image.fromarray(dilated_binary_img.astype(np.uint8) * 255)
    return dilated_mask, dilated_binary_img


def show_masks(img_np, masks, alpha=0.5):
    np.random.seed(0)
    img_np_copy = copy.deepcopy(img_np)
    for mask in masks:
        color = np.random.randint(0, 255, 3)
        img_np_copy[mask][:, :3] = img_np_copy[mask][:, :3] * (1 - alpha) + color * alpha
    return img_np_copy.astype(np.uint8)

def show_boxes(img_np, boxes, color=(255, 0, 0, 255), thickness=2, show_index=False):
    if boxes is None:
        return img_np
    img_np_copy = copy.deepcopy(img_np)
    for idx, box in enumerate(boxes):
        x, y, w, h = box
        cv2.rectangle(img_np_copy, (x, y), (w, h), color, thickness)
        if show_index:
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = str(idx)
            textsize = cv2.getTextSize(text, font, 1, 2)[0]
            cv2.putText(img_np_copy, text, (x, y+textsize[1]), font, 1, color, thickness)
    
    return img_np_copy

def create_mask_output(img_np, masks, box_filters):
    msg = f"Creating mask output..."

    img_masked, masks_gallery, img_matted = [], [], []
    box_filters = box_filters.astype(int) if box_filters is not None else None
    for mask in masks:
        masks_gallery.append(Image.fromarray(np.any(mask, axis=0)))

        blended_image = show_masks(show_boxes(img_np, box_filters), mask)
        img_masked.append(Image.fromarray(blended_image))

        img_np[~np.any(mask, axis=0)] = np.zeros(len(img_np.shape))
        img_matted.append(Image.fromarray(img_np))

    return (img_masked, masks_gallery, img_matted), None


def create_mask_output_and_save(
    file_name, 
    save_dir, 
    img_np, 
    masks, 
    box_filters, 
    dilation_amt, 
    save_image, 
    save_mask, 
    save_image_background,
    save_image_blend, 
    save_image_masked
    ):
    msg = f"Creating mask batch output..."
    # check if save dir exists and overwrite if exists
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    filename, ext = os.path.splitext(os.path.basename(file_name))
    # ext = ".png" # JPEG not compatible with RGBA
    merged_masks = []
    for idx, mask in enumerate(masks):

        blended_image = show_masks(show_boxes(img_np, box_filters), mask)
        merged_mask = np.any(mask, axis=0)

        if dilation_amt:
            _, merged_mask = dilate_mask(merged_mask, dilation_amt)

        if save_image:
            output_image = Image.fromarray(img_np)
            output_image.save(os.path.join(save_dir, f"{filename}_{idx}_original{ext}"))
        if save_mask:
            output_mask = Image.fromarray(merged_mask)
            output_mask.save(os.path.join(save_dir, f"{filename}_{idx}_mask{ext}"))
        if save_image_background:
            background_mask = ~merged_mask
            output_mask = Image.fromarray(background_mask)
            output_mask.save(os.path.join(save_dir, f"{filename}_{idx}_background{ext}"))
        if save_image_blend:
            output_blend = Image.fromarray(blended_image)
            output_blend.save(os.path.join(save_dir, f"{filename}_{idx}_blend{ext}"))
        if save_image_masked:
            output_matted = img_np.copy()
               # check if there is an alpha channel
            if output_matted.shape[2] == 4:
                # set alpha to 0 for the region to be transparent
                output_matted[~merged_mask, 3] = 0
            else:
                # if no alpha channel exists, create one
                h, w = output_matted.shape[:2]
                alpha = np.ones((h, w)) * 255
                alpha[~merged_mask] = 0
                output_matted = np.dstack([output_matted, alpha]) 

            output_matted = Image.fromarray(output_matted.astype(np.uint8))
            output_matted.save(os.path.join(save_dir, f"{filename}_{idx}_matted.png"))
        
        merged_masks.append(merged_mask)
    return merged_masks, msg

def sam_predict(sam_model_type, img_np, box_filters=None, positive_points=[], negative_points=[],):
    print("Start SAM Processing")
    # check if sam_model_dir exists
    sam, msg = load_sam_model(sam_model_type)
    if not sam:
        return msg
    else: 
        print(msg)
    
    img_np_rgb = img_np[..., :3]
    print(f"Image shape: {img_np.shape}")
    predictor = SamPredictor(sam)
    predictor.set_image(img_np_rgb)
    
    if box_filters.shape[0]>1:
        print(f"SAM running with {box_filters.shape[0]} boxes, discard point prompts")
        boxes_transformed = predictor.transform.apply_boxes_torch(box_filters, img_np_rgb.shape[:2])
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            boxes=boxes_transformed.to(device),
            multimask_output=True
            )
        masks = masks.permute(1,0,2,3).cpu().numpy()
    else:
        num_boxes = 0 if box_filters is None else box_filters.shape[0]
        num_points = len(positive_points) + len(negative_points)
        if not num_boxes and  not num_points:
            return False, "Please provide at least one point or text prompt"
        print(f"SAM running with {num_boxes} boxes, {len(positive_points)} positive points and {len(negative_points)} negative points")
        point_coords = torch.cat([positive_points, negative_points], dim=0) if num_points else None
        point_labels = torch.cat([torch.ones(len(positive_points)), torch.zeros(len(negative_points))], dim=0) if num_points else None
        masks, _, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box_filters[0],
            multimask_output=True
        )
        masks = masks[:, None, ...]
    
    (img_masked, masks_gallery, img_matted), msg =  create_mask_output(img_np, masks, box_filters)
    # save img_masked, masks_gallery, img_matted to files
    img_masked[0].save("/root/fstudio/sam_output/img_masked.png")
    masks_gallery[0].save("/root/fstudio/sam_output/masks_gallery.png")
    img_matted[0].save("/root/fstudio/sam_output/img_matted.png")
    return img_masked, masks_gallery, img_matted

# if __name__ == "__main__":

#     sam_model_type = "H"
#     sam, msg = load_sam_model(sam_model_type)

#     img_list, filename_list, msg = load_img_from_path("/root/autodl-tmp/sd-dataset/alex/alex-768")
#     if not img_list:
#         print(msg)
    
    
#     # boxes = dino_predict(dino_model_type, images[0], "t-shirt", 2, 0.4)
#     # print(boxes)