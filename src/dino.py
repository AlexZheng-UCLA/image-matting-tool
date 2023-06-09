import cv2
import torch
import os
import numpy as np
from PIL import Image
from collections import OrderedDict

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
dino_model_cache = OrderedDict()
dino_model_dir = "/root/autodl-tmp/grounding-dino"
dino_model = {
    "T": {
        "name": "GroundingDINO_SwinT_OGC",
        "checkpoint": os.path.join(dino_model_dir, "groundingdino_swint_ogc.pth"),
        "config": os.path.join(dino_model_dir, "GroundingDINO_SwinT_OGC.cfg.py"),
        "url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth",
    },
    "B": {
        "name": "GroundingDINO_SwinB_COGCOOR",
        "checkpoint": os.path.join(dino_model_dir, "groundingdino_swinb_cogcoor.pth"),
        "config": os.path.join(dino_model_dir, "GroundingDINO_SwinB.cfg.py"),
        "url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth"
    },
}

def show_boxes(img_np, boxes, color=(255, 0, 0, 255), thickness=2, show_index=False):
    if boxes is None:
        return img_np

    for idx, box in enumerate(boxes):
        x, y, w, h = box
        cv2.rectangle(img_np, (x, y), (w, h), color, thickness)
        if show_index:
            cv2.putText(img_np, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return img_np

def load_dino_model(dino_model_type):
    msg = f"Initializing {dino_model[dino_model_type]['name']} ..."
    if dino_model_type in dino_model_cache:
        msg += "\n dino_checkpoint in dino_model_cache"
        dino = dino_model_cache[dino_model_type]
        dino.to(device=device)
    else:
        msg += f"\n load dino model from {dino_model_dir}"
        from groundingdino.util.inference import load_model
        dino = load_model(dino_model[dino_model_type]['config'], dino_model[dino_model_type]['checkpoint'])
        dino.to(device=device)
        dino_model_cache[dino_model_type] = dino
    dino.eval()
    return dino, msg

def load_dino_image(image_pil):
    import groundingdino.datasets.transforms as T
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image

def get_grounding_output(model, image, caption, num_boxes, box_threshold):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)

    # get the index of max num_boxes
    max_logits = logits.max(dim=1)[0]
    mask = torch.zeros_like(max_logits, dtype=torch.bool)
    for i in range(int(num_boxes)):
        if max_logits[i] > box_threshold:
            mask[i] = 1
    
    logits_filtered = logits[mask]  # num_filt, 256
    boxes_filtered = boxes[mask]  # num_filt, 4

    return boxes_filtered.cpu()


def dino_predict(dino_model_type, input_image, text_prompt, num_boxes, box_threshold):
    dino, msg = load_dino_model(dino_model_type)
    if not dino:
        return False, msg
    
    image = load_dino_image(input_image.convert("RGB"))
    boxes = get_grounding_output(
        dino, image, text_prompt, num_boxes, box_threshold
    )

    H, W = input_image.size[1], input_image.size[0]
    for i in range(boxes.size(0)):
        boxes[i] = boxes[i] * torch.Tensor([W, H, W, H])
        boxes[i][:2] -= boxes[i][2:] / 2
        boxes[i][2:] += boxes[i][:2]

    return boxes, "successfully run GroundingDINO"


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dino_model_type = "T"
    dino, msg = load_dino_model(dino_model_type)

    images, msg = load_img_from_path("/root/autodl-tmp/sd-dataset/alex/alex-768")
    if not images:
        print(msg)
    
    
    boxes = dino_predict(dino_model_type, images[0], "t-shirt", 2, 0.4)
    print(boxes)