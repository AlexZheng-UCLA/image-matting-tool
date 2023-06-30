from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from typing import List
import pika
import json
import uuid
import os
import shutil
import aiofiles

app = FastAPI()

@app.post("/start_process/")
async def start_process(background_tasks: BackgroundTasks,
                        sam_model_type: str,
                        dino_model_type: str,
                        text_prompt: str,
                        num_boxes: int,
                        box_threshold: float,
                        dilation_amt: float,
                        background_dir: str,
                        save_dir: str,
                        multimask_output: bool,
                        save_image: bool,
                        save_mask: bool,
                        save_background: bool,
                        save_blend: bool,
                        save_image_matted: bool,
                        save_image_pasted: bool,
                        images: List[UploadFile] = File(...)):
    
    img_source_dir = f"{save_dir}/images/"
    os.makedirs(img_source_dir, exist_ok=True)
    
    for image in images:
        async with aiofiles.open(f"{img_source_dir}/{image.filename}", 'wb') as out_file:
            content = await image.read()  # async read
            await out_file.write(content)  # async write
    
    # Generate a unique id for the job
    job_id = str(uuid.uuid4())
    
    # Prepare the job data
    job_data = {
        "job_id": job_id,
        "sam_model_type": sam_model_type,
        "dino_model_type": dino_model_type,
        "text_prompt": text_prompt,
        "num_boxes": num_boxes,
        "box_threshold": box_threshold,
        "dilation_amt": dilation_amt,
        "img_source_dir": img_source_dir,
        "background_dir": background_dir,
        "save_dir": save_dir,
        "multimask_output": multimask_output,
        "save_image": save_image,
        "save_mask": save_mask,
        "save_background": save_background,
        "save_blend": save_blend,
        "save_image_matted": save_image_matted,
        "save_image_pasted": save_image_pasted,
    }
    
    # Add the job to the RabbitMQ queue
    background_tasks.add_task(add_job_to_queue, job_data)
    
    # Return the job_id to the client
    return {"job_id": job_id}

def add_job_to_queue(job_data):
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()

    channel.queue_declare(queue='job_queue')

    channel.basic_publish(exchange='', routing_key='job_queue', body=json.dumps(job_data))

    connection.close()
