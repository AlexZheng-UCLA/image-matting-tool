from src.function import *


sam_model_type = "H"
dino_model_type = "T"
dataset = "xxy"
background = "resorts"
img_source_dir = os.path.join("/root/autodl-tmp/dataset", dataset)
background_dir = os.path.join("/root/autodl-tmp/dataset", background)
save_dir = os.path.join("/root/image-matting-tool", "outputs", dataset)

text_prompt = "a person"
num_boxes = 1
box_threshold = 0.5

dilation_amt = 0
multimask_output = True
save_image = True
save_mask = True
save_background = True
save_blend = True
save_image_masked = True
h_shift = True


full_process(
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
    save_image_bg=True,
    h_shift=True,
)


# merged_masks, process_info = matting(
#     sam_model_type,dino_model_type,text_prompt,num_boxes,box_threshold,dilation_amt,img_source_dir,save_dir,
#     multimask_output,save_image,save_mask,save_background,save_blend,save_image_masked,
# )
# print(process_info)


# merged_masks, process_info = matting(
#     sam_model_type,dino_model_type,text_prompt,num_boxes,box_threshold,dilation_amt,img_source_dir,save_dir,
#     multimask_output,save_image,save_mask,save_background,save_blend,save_image_masked,
# )
# print(process_info)

# background_dir = os.path.join("/root/image-matting-tool", "backgrounds")
# mask_option = "2"  # ["first", "1", "2", "3", "largest", "smallest", "merge"]
# h_shift = True

# _, process_info = change_background(
#     img_source_dir,save_dir,background_dir,
#     merged_masks,mask_option,h_shift,
# )
# print(process_info)
