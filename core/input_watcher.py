import json
import logging
import os
import shutil
import threading
import time
import uuid
import zipfile
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from core.interrogator import Interrogator

logger = logging.getLogger(__name__)


class InputWatcher(FileSystemEventHandler):
    def __init__(self, output_path: str, working_path: str, interrogator: Interrogator):
        self._output_path = output_path
        self._working_path = working_path
        self._interrogator = interrogator

    def clean_start(self):
        delete_all_in_path(self._working_path)

    def reprocess_unhandled_jobs(self, input_path):
        files_by_oldest = list_files_sorted_by_oldest(input_path)
        for file in files_by_oldest:
            file_path = os.path.join(input_path, file)
            thread: threading.Thread = threading.Thread(
                target=self._handle_path,
                args=(file_path,)
            )
            thread.start()

    def on_created(self, event):
        os.makedirs(self._output_path, exist_ok=True)
        print(f"File created: {event.src_path}")
        zip_path = event.src_path
        self._handle_path(zip_path)

    def _handle_path(self, zip_path):
        job_id = self._zip_path_to_job_id(zip_path)
        try:
            self._validate_zip_file(zip_path)
            self._wait_until_file_ready(zip_path)
        except ValueError as e:
            self._write_error_response(str(e), job_id)
            logging.error(f"Failed to validate zip file: {e}")
        except TimeoutError as e:
            self._write_error_response(str(e), job_id)
            logging.error(f"Timeout waiting for zip file {zip_path}: {e}")
        try:
            self._handle_zip(zip_path)
        except ValueError as e:
            self._write_error_response(str(e), job_id)
            logging.error(f"Failed to handle zip file: {e}")
        except RuntimeError as e:
            self._write_error_response(str(e), job_id)
            logging.error(f"Failed to handle zip file: {e}")
        except TimeoutError as e:
            self._write_error_response(str(e), job_id)
            logging.error(f"Timeout waiting for zip file {zip_path}: {e}")
        os.remove(zip_path)
        job_working_dir = self.get_job_working_dir(job_id)
        shutil.rmtree(job_working_dir)
        logging.info(f"deleted working dir: {job_working_dir}")

    def _validate_zip_file(self, zip_path):
        os.path.basename(zip_path)
        file_name, extension = os.path.splitext(zip_path)
        if extension != ".zip":
            raise ValueError(f"File {zip_path} must have .zip extension")

    def _wait_until_file_ready(self, zip_path):
        return self._wait_until_stable(zip_path)
        # max_wait = 5
        # expiry = time.time() + max_wait
        # while not _is_file_closed(zip_path):
        #     if time.time() > expiry:
        #         raise TimeoutError(f"File {zip_path} wasn't closed closed after {max_wait} seconds.")
        #     time.sleep(0.1)
    def _wait_until_stable(self, zip_path):
        previous_size = -1
        max_wait = 5
        expiry = time.time() + max_wait
        while True:
            if time.time() > expiry:
                raise TimeoutError(f"File {zip_path} wasn't ready after {max_wait} seconds.")
            current_size = os.path.getsize(zip_path)
            if current_size == previous_size:
                return True
            previous_size = current_size
            time.sleep(1)
    def _handle_zip(self, zip_path):
        logging.info(f"Handling zip file: {zip_path}")
        job_id: str = self._zip_path_to_job_id(zip_path)
        job_working_dir = self.get_job_working_dir(job_id)
        os.makedirs(job_working_dir, exist_ok=True)
        unzip_file(zip_path, job_working_dir)
        self._start_job(job_id)
        logging.info(f"deleted input zip file: {zip_path}")

    def _zip_path_to_job_id(self, zip_path) -> str:
        job_id = os.path.splitext(os.path.basename(zip_path))[0]
        return job_id

    def _write_error_response(self, response, job_id):
        job_response = {
            "error": response,
        }
        response_file_name = f"{job_id}.json"
        path = os.path.join(self._output_path, response_file_name)
        with open(path, "w") as f:
            logging.info(f"Wrote error {response_file_name}")
            json.dump(job_response, f, indent=4)

    def get_job_working_dir(self, id):
        job_working_dir = os.path.join(self._working_path, id)
        return job_working_dir

    def _start_job(self, job_id: str):
        job_working_dir = self.get_job_working_dir(job_id)
        images = self.find_images(job_id)
        if len(images) == 0:
            raise ValueError(f"Job {job_id} has no images")
        image_path = os.path.join(job_working_dir, images[0])
        job_file_path = os.path.join(job_working_dir, "job.json")
        if not os.path.exists(job_file_path):
            raise ValueError(f"Job {job_id} has no job file")
        job_spec = read_json(job_file_path)
        # need a model name
        if "model_name" not in job_spec:
            raise ValueError(f"Job {job_id} has no model name")
        model_name = job_spec["model_name"]
        if not self._is_valid_model(model_name):
            raise ValueError(f"Job {job_id} has invalid model name: {model_name}")
        try:
            tags = self._interrogator.process(image_path, model_name)
        except ValueError as e:
            # we're not using it properly.
            # todo: cleanup
            return
        logging.info(f"got tags: {tags}")
        logging.info(f"finished job {job_id} with model name: {model_name}")
        job_response = {
            "job_id": job_id,
            "model": model_name,
            "tags": tags,
        }
        response_file_name = f"{job_id}.json"
        path = os.path.join(self._output_path, response_file_name)
        with open(path, "w") as f:
            logging.info(f"Wrote {response_file_name}")
            json.dump(job_response, f, indent=4)

    def _is_valid_model(self, model_name) -> bool:
        valid_models = Interrogator.get_valid_models()
        if model_name not in valid_models:
            return False
        return True

    def find_images(self, job_id):
        job_working_dir = self.get_job_working_dir(job_id)
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
        directories = list_directories(path)
        for d in directories:
            rm_path = os.path.join(path, d)
            shutil.rmtree(rm_path)
        print(f"Deleted all contents of {path}")
    else:
        print(f"Path {path} does not exist.")


def unzip_file(zip_file_path, destination_folder):
    # Ensure the destination folder exists
    os.makedirs(destination_folder, exist_ok=True)
    expiry = time.time() + 5

    # Open the zip file
    while True:
        if time.time() > expiry:
            os.remove(zip_file_path)
            raise TimeoutError
        try:
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                # Extract all files to the destination folder
                zip_ref.extractall(destination_folder)
                print(f"Extracted all files to {destination_folder}")
                return
        except zipfile.BadZipFile as e:
            print(f"Failed to extract {zip_file_path}: {e}")
            time.sleep(0.1)



def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def list_directories(path):
    # List only directories in the given path
    return [entry for entry in os.listdir(path) if os.path.isdir(os.path.join(path, entry))]

def list_files_sorted_by_oldest(directory):
    # Get all file paths in the directory
    files = [os.path.join(directory, file) for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]
    # Sort files by modified time (oldest first)
    files.sort(key=lambda x: os.path.getmtime(x))
    return files