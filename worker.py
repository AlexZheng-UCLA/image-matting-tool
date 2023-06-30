import pika
import json
from src.function import *

def worker_callback(ch, method, properties, body):
    # Parse job data
    job_data = json.loads(body)

    # Call full_process with job data
    full_process(
        job_data["sam_model_type"], 
        job_data["dino_model_type"], 
        job_data["text_prompt"], 
        job_data["num_boxes"],
        job_data["box_threshold"], 
        job_data["dilation_amt"],
        job_data["img_source_dir"], 
        job_data["background_dir"],
        job_data["save_dir"],
        job_data["multimask_output"],
        job_data["save_image"], 
        job_data["save_mask"], 
        job_data["save_background"],
        job_data["save_blend"], 
        job_data["save_image_matted"],
        job_data["save_image_pasted"],
    )
    
    # Acknowledge message
    ch.basic_ack(delivery_tag = method.delivery_tag)

# Set up RabbitMQ connection
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

# Declare the queue
channel.queue_declare(queue='job_queue') # job_queue is the name of the queue

# Start consuming messages
channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue='job_queue', on_message_callback=worker_callback)
channel.start_consuming()
