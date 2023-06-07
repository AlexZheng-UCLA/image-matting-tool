import torch
import os 
from PIL import Image
import numpy as np
import cv2
from scipy.ndimage import binary_dilation
from segment_anything import SamPredictor
from src.dino import dino_predict
from src.sam import load_sam_model, create_mask_output_and_save
from src.image import load_img_from_path, load_background_from_path, move_masked_add_background
from src.util import garbage_collect
import warnings

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

def full_process(
    sam_model_type, 
    dino_model_type, 
    text_prompt, 
    num_boxes,
    box_threshold, 
    dilation_amt,
    img_source_dir, 
    background_dir,
    save_dir,
    multimask_output=True,
    mask_option="1",
    save_image=True, 
    save_mask=True, 
    save_background=True,
    save_blend=True, 
    save_image_masked=True,
    save_process=True,
    ):

    if text_prompt is None or text_prompt == "":
        return "Please add text prompts to generate masks"
    print("Start groundingdino + sam processing")
    # get the parent folder of save_dir 
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    process_info = ""
    sam, msg = load_sam_model(sam_model_type)
    if not sam:
        return msg
    else:
        process_info += (msg + "\n") if msg else ""

    predictor = SamPredictor(sam)
    img_list, filename_list, msg = load_img_from_path(img_source_dir)
    if not img_list:
        return msg
    else:
        process_info += (msg + "\n") if msg else ""

    for idx, img in enumerate(img_list):
        img_np = np.array(img)
        img_np_rgb = img_np[..., :3]

        # grounding-dino ********************** 
        box_filters, msg = dino_predict(dino_model_type, img, text_prompt, num_boxes, box_threshold)

        if box_filters is None or box_filters.shape[0] == 0:
            msg = f"GroundingDINO generated 0 box for image {filename_list[idx]}, please lower the box threshold if you want any segmentation for this image. "
            process_info += (msg + "\n") if msg else ""
            continue
        # sam **********************************
        predictor.set_image(img_np_rgb)
        transformed_boxes = predictor.transform.apply_boxes_torch(box_filters, img_np.shape[:2])
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(device),
            multimask_output=multimask_output,
            )
        masks = masks.permute(1, 0, 2, 3).cpu().numpy()
        box_filters = box_filters.cpu().numpy().astype(int)
        
        # post-process **************************
        merged_masks, msg = create_mask_output_and_save(
            filename_list[idx], save_dir, img_np, masks, box_filters, dilation_amt, save_image, save_mask, save_background, save_blend, save_image_masked)
        if merged_masks is None:
            return msg
        else:
            process_info += (msg + "\n") if msg else ""

        background_list, img = load_background_from_path(background_dir)
        if background_list is None:
            return msg
        else:
            process_info += (msg + "\n") if msg else ""

        img_processed, msg = move_masked_add_background(
            filename_list[idx], save_dir, img_np, background_list, merged_masks, mask_option, save_process)

    garbage_collect(sam)
    return process_info + "Done"

def matting(
    sam_model_type, 
    dino_model_type, 
    text_prompt, 
    num_boxes,
    box_threshold, 
    dilation_amt,
    img_source_dir, 
    save_dir,
    multimask_output=True,
    save_image=True, 
    save_mask=True, 
    save_background=True,
    save_blend=True, 
    save_image_masked=True,
):
    if text_prompt is None or text_prompt == "":
        return "Please add text prompts to generate masks"
    print("Start groundingdino + sam processing")
    # get the parent folder of save_dir 
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    process_info = ""
    sam, msg = load_sam_model(sam_model_type)
    if not sam:
        return msg
    else:
        process_info += (msg + "\n") if msg else ""

    predictor = SamPredictor(sam)
    img_list, filename_list, msg = load_img_from_path(img_source_dir)
    if not img_list:
        return msg
    else:
        process_info += (msg + "\n") if msg else ""

    for idx, img in enumerate(img_list):
        img_np = np.array(img)
        img_np_rgb = img_np[..., :3]

        # grounding-dino ********************** 
        box_filters, msg = dino_predict(dino_model_type, img, text_prompt, num_boxes, box_threshold)

        if box_filters is None or box_filters.shape[0] == 0:
            msg = f"GroundingDINO generated 0 box for image {filename_list[idx]}, please lower the box threshold if you want any segmentation for this image. "
            process_info += (msg + "\n") if msg else ""
            continue
        # sam **********************************
        predictor.set_image(img_np_rgb)
        transformed_boxes = predictor.transform.apply_boxes_torch(box_filters, img_np.shape[:2])
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(device),
            multimask_output=multimask_output,
            )
        masks = masks.permute(1, 0, 2, 3).cpu().numpy()
        box_filters = box_filters.cpu().numpy().astype(int)
        
        # post-process **************************
        merged_masks, msg = create_mask_output_and_save(
            filename_list[idx], save_dir, img_np, masks, box_filters, dilation_amt, save_image, save_mask, save_background, save_blend, save_image_masked)
        if merged_masks is None:
            return msg
        else:
            process_info += (msg + "\n") if msg else ""

    return merged_masks, process_info

def change_background(
    img_source_dir, 
    save_dir, 
    background_dir, 
    merged_masks,
    mask_option, 
    h_shift,
):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    process_info = ""
    img_list, filename_list, msg = load_img_from_path(img_source_dir)
    if not img_list:
        return msg
    else:
        process_info += (msg + "\n") if msg else ""

    background_list, img = load_background_from_path(background_dir)
    if background_list is None:
        return msg
    else:
        process_info += (msg + "\n") if msg else ""

    for idx, img in enumerate(img_list):
        img_np = np.array(img)
        img_np_rgb = img_np[..., :3]
        img_processed, msg = move_masked_add_background(
            filename_list[idx], save_dir, img_np, background_list, merged_masks, mask_option, h_shift, True)

    return img_processed, process_info


if __name__ == "__main__":

    sam_model_type = "H"
    dino_model_type = "T"
    dataset = "chenweiting-768"
    img_source_dir = os.path.join("/root/autodl-tmp/dataset", dataset)
    background_dir = os.path.join("/root/image-matting-tool", "backgrounds")
    save_dir = os.path.join("/root/image-matting-tool", "output")

    text_prompt = "a person"
    num_boxes = 1
    box_threshold = 0.5
    dilation_amt = 0

    multimask_output = True
    mask_option = "largest"  # ["first", "1", "2", "3", "largest", "smallest", "merge"]
    save_image = True
    save_mask = True
    save_background = False
    save_blend = False
    save_image_masked = True
    save_process = True

    process_info = full_process(
        sam_model_type=sam_model_type, 
        dino_model_type=dino_model_type, 
        text_prompt=text_prompt, 
        num_boxes=num_boxes,
        box_threshold=box_threshold, 
        dilation_amt=dilation_amt,
        img_source_dir=img_source_dir, 
        background_dir = background_dir,
        save_dir=save_dir,
        multimask_output=multimask_output,
        mask_option=mask_option,
        save_image=save_image, 
        save_mask=save_mask, 
        save_background=save_background,
        save_blend=save_blend, 
        save_image_masked=save_image_masked,
        save_process = save_process
    )
    print(process_info)
    # masks = mask_entire_image(sam, images[0])