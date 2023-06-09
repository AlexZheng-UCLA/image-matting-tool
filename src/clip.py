from PIL import Image
import requests
import cv2
import torch
import os
import numpy as np
from PIL import Image
from collections import OrderedDict
from transformers import CLIPProcessor, CLIPModel


device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
clip_model_cache = OrderedDict()
clip_model_dir = "/root/autodl-tmp/clip"
clip_model = {
    "L": {
        "name": "vit_l",
        "folder": "clip-vit-large"
    }
}

def load_clip_model(model_type):
    if model_type is None:
        return None, None, "Please provide clip model!"
    if not model_type in clip_model:
        return None, None, "Provide correct clip model type"
    
    if model_type in clip_model_cache:
        clip = clip_model_cache[model_type]
        processor = clip_model_cache[f"{model_type}_processor"]
        msg = f"load clip {model_type} from clip_model_cache"
    else:
        model_path = os.path.join(clip_model_dir, clip_model[model_type]["folder"])
        print(model_path)
        clip = CLIPModel.from_pretrained(model_path)
        processor = CLIPProcessor.from_pretrained(model_path)
        msg = f"load clip {clip_model[model_type]} from {model_path}"
        clip_model_cache[model_type] = clip
        clip_model_cache[f"{model_type}_processor"] = processor

    clip.to(device=device)
    clip.eval()
    return clip, processor, msg


def closest_image(text, img_list, clip, processor):
    inputs = processor(text=text, images=img_list, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = clip(**inputs)
    logits_per_text = outputs.logits_per_text # this is the image-text similarity score
    probs = logits_per_text.softmax(dim=1) # we can take the softmax to get the label probabilities
    idx_closest = probs.argmax(dim=1) # we can take the softmax to get the label probabilities

    return idx_closest, probs

if __name__ == "__main__":
    clip, processor, _ = load_clip_model("L")
    img_list,_, _, = load_img_from_path("/root/sd-dataset/yml")
    idx, probs = closest_image("a photo of a cat", img_list, clip, processor)
    print(idx)
    print(probs)


# Contrastive loss for image-text similarity.
# logits_per_image:(:obj:`torch.FloatTensor` of shape :obj:`(image_batch_size, text_batch_size)`):
#     The scaled dot product scores between :obj:`image_embeds` and :obj:`text_embeds`. This represents the
#     image-text similarity scores.
# logits_per_text:(:obj:`torch.FloatTensor` of shape :obj:`(text_batch_size, image_batch_size)`):
#     The scaled dot product scores between :obj:`text_embeds` and :obj:`image_embeds`. This represents the
#     text-image similarity scores.
# text_embeds(:obj:`torch.FloatTensor` of shape :obj:`(batch_size, output_dim`):
#     The text embeddings obtained by applying the projection layer to the pooled output of
#     :class:`~transformers.CLIPTextModel`.
# image_embeds(:obj:`torch.FloatTensor` of shape :obj:`(batch_size, output_dim`):
#     The image embeddings obtained by applying the projection layer to the pooled output of
#     :class:`~transformers.CLIPVisionModel`.
# text_model_output(:obj:`BaseModelOutputWithPooling`):
#     The output of the :class:`~transformers.CLIPTextModel`.
# vision_model_output(:obj:`BaseModelOutputWithPooling`):
#     The output of the :class:`~transformers.CLIPVisionModel`.