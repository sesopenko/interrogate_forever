import json
import logging
import os
import shutil
import tempfile
import time
import uuid
import zipfile
from io import BytesIO

import click

logger = logging.getLogger(__name__)

@click.command

@click.option("--image-path", required=True, help="Path to the image")
@click.option("--model-name", default="Salesforce/blip-image-captioning-base", required=True, help="Name of the model to use, ie: 'openai/clip-vit-large-patch14'")
def create_job(image_path: str, model_name: str):
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)
    file_name = os.path.basename(image_path)
    job_id = str(uuid.uuid4())
    job_spec = {
        "model_name": model_name,
        "job_id": job_id,
        "input_image_filename": file_name,
    }

    zip_file_folder = os.path.join(os.getcwd(), 'data', 'input', f"{job_id}.zip")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
        with zipfile.ZipFile(tmp, 'w') as zip:
            zip.write(image_path, arcname=file_name)

            # add the json string
            json_string = json.dumps(job_spec, indent=2)
            json_bytes = BytesIO(json_string.encode('utf-8'))
            zip.writestr("job.json", json_bytes.read())
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    shutil.move(tmp_name, zip_file_folder)
    logging.info(f"Created job {job_id} in {zip_file_folder}")