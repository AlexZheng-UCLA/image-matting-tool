from src.function import *


sam_model_type = "H"
dino_model_type = "T"
dataset = "chenweiting-512"
img_source_dir = os.path.join("/root/autodl-tmp/dataset", dataset)
save_dir = os.path.join("/root/image-matting-tool", "output", dataset)

text_prompt = "a person"
num_boxes = 1
box_threshold = 0.5
dilation_amt = 0

multimask_output = True
save_image = True
save_mask = True
save_background = False
save_blend = False
save_image_masked = True

merged_masks, process_info = matting(
    sam_model_type,dino_model_type,text_prompt,num_boxes,box_threshold,dilation_amt,img_source_dir,save_dir,
    multimask_output,save_image,save_mask,save_background,save_blend,save_image_masked,
)
print(process_info)


background_dir = os.path.join("/root/image-matting-tool", "backgrounds")
mask_option = "largest"  # ["first", "1", "2", "3", "largest", "smallest", "merge"]
h_shift = True

_, process_info = change_background(
    img_source_dir,save_dir,background_dir,
    merged_masks,mask_option,h_shift,
)
print(process_info)
