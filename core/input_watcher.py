import json
import logging
import os
import shutil
import time
import uuid
import zipfile
from pathlib import Path

from watchdog.events import FileSystemEventHandler

logger = logging.getLogger(__name__)


class InputWatcher(FileSystemEventHandler):
    def __init__(self, output_path: str, working_path: str):
        self._output_path = output_path
        self._working_path = working_path

    def clean_start(self):
        delete_all_in_path(self._working_path)

    def on_created(self, event):
        os.makedirs(self._output_path, exist_ok=True)
        print(f"File created: {event.src_path}")
        zip_path = event.src_path
        try:
            self._validate_zip_file(zip_path)
            self._wait_until_file_ready(zip_path)
        except ValueError as e:
            logging.error(f"Failed to validate zip file: {e}")
        try:
            self._handle_zip(zip_path)
        except ValueError as e:
            logging.error(f"Failed to handle zip file: {e}")
        except RuntimeError as e:
            logging.error(f"Failed to handle zip file: {e}")

    def _validate_zip_file(self, zip_path):
        os.path.basename(zip_path)
        file_name, extension = os.path.splitext(zip_path)
        if extension != ".zip":
            raise ValueError(f"File {zip_path} must have .zip extension")
        if not is_valid_uuid(file_name):
            raise ValueError(f"File {zip_path} must have a uuid file name")

    def _wait_until_file_ready(self, zip_path):
        max_wait = 5
        expiry = time.time() + max_wait
        while not _is_file_closed(zip_path):
            if time.time() > expiry:
                raise TimeoutError(f"File {zip_path} wasn't closed closed after {max_wait} seconds.")
            time.sleep(0.1)

    def _handle_zip(self, zip_path):
        logging.info(f"Handling zip file: {zip_path}")
        job_id, extension = os.path.splitext(os.path.basename(zip_path))
        job_working_dir = self.get_job_working_dir(job_id)
        os.makedirs(job_working_dir, exist_ok=True)
        unzip_file(zip_path, job_working_dir)
        self._start_job(job_id)

    def get_job_working_dir(self, id):
        job_working_dir = os.path.join(self._working_path, id)
        return job_working_dir

    def _start_job(self, job_id: str):
        job_working_dir = self.get_job_working_dir(id)
        images = self.find_images(job_id)
        if len(images) == 0:
            raise ValueError(f"Job {job_id} has no images")
        image_path = images[0]
        job_file_path = os.path.join(job_working_dir, "job.json")
        if not os.path.exists(job_file_path):
            raise ValueError(f"Job {job_id} has no job file")
        job_spec = read_json(job_file_path)
        # need a model name
        if "model_name" not in job_spec:
            raise ValueError(f"Job {job_id} has no model name")
        model_name = job_spec["model_name"]
        if not self._is_valid_model_name(model_name):
            raise ValueError(f"Job {job_id} has invalid model name: {model_name}")


    def _is_valid_model(self, model_name) -> bool:
        valid_models = [
            "openai/clip-vit-large-patch14",
        ]
        if model_name not in valid_models:
            return False
        return True


    def find_images(self, job_id):
        job_working_dir = self.get_job_working_dir(id)
        try:
            return [
                f.name
                for f in Path(job_working_dir).iterdir()
                if f.is_file() and f.suffix.lower() in {'.jpg', '.jpeg', '.png'}
            ]
        except RuntimeError:
            logging.error(f"Directory not found: {job_working_dir}")
            return []

        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return []

    def _supported_extensions(self):
        return [".jpg", ".jpeg", '.png"']


def _is_file_closed(file_path):
    try:
        # Try to open the file for exclusive access
        with open(file_path, 'rb') as f:
            pass
        return True
    except (OSError, IOError):
        # File is still being written
        return False


def delete_all_in_path(path):
    if os.path.exists(path):
        shutil.rmtree(path)  # Recursively deletes all files and folders
        os.makedirs(path, exist_ok=True)  # Recreate the empty directory if needed
        print(f"Deleted all contents of {path}")
    else:
        print(f"Path {path} does not exist.")


def unzip_file(zip_file_path, destination_folder):
    # Ensure the destination folder exists
    os.makedirs(destination_folder, exist_ok=True)

    # Open the zip file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # Extract all files to the destination folder
        zip_ref.extractall(destination_folder)
        print(f"Extracted all files to {destination_folder}")


def is_valid_uuid(uuid_string):
    try:
        # Attempt to create a UUID object from the string
        uuid_obj = uuid.UUID(uuid_string, version=4)
        return str(uuid_obj) == uuid_string.lower()
    except ValueError:
        # Raised if the string is not a valid UUID
        return False

def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data