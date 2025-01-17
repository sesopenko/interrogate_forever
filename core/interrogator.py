import logging
from pathlib import Path

import torch
from onnxruntime import InferenceSession

import numpy as np
from PIL import Image
from transformers import Blip2Processor, \
    Blip2ForConditionalGeneration, \
    BlipProcessor, BlipForConditionalGeneration, \
    AutoModelForImageClassification, \
    AutoFeatureExtractor, AutoConfig
from huggingface_hub import hf_hub_download
import pandas as pd

from core import dbimutils as dbimutils


from typing import Optional
import threading

logger = logging.getLogger(__name__)

ARCHITECTURE_VIT = "vit"
ARCHITECTURE_BLIP = "blip"
ARCHITECTURE_BLIP2 = "blip2"


class Interrogator:
    def __init__(self):
        self._current_model_name: Optional[str] = None
        self._current_model: Optional[Blip2ForConditionalGeneration | BlipForConditionalGeneration] = None
        self._processor: Optional[Blip2Processor | BlipProcessor] = None
        self._feature_extractor: Optional[AutoFeatureExtractor] = None
        self._mutex: threading.Lock = threading.Lock()
        self._providers: list[str] = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self._model_tags = None

    def process(self, image_path: str, model_name: str) -> list[str]:
        logging.info(f"Processing {image_path} with model {model_name}")
        with self._mutex:
            if model_name not in Interrogator.get_valid_models():
                raise ValueError(f"Invalid model: {model_name}")
            if self._current_model_name == None:
                # first run, set it up.
                logging.info("No current model, setting up")
                self._setup_model(model_name)
            elif model_name != self._current_model_name:
                logging.info("Changing model, tearing down model")
                self._teardown_model()
                logging.info(f"Changing model to {model_name}")
                self._setup_model(model_name)


            # prepare inputs for the model
            architecture = Interrogator.get_model_architecture(model_name)
            if architecture == ARCHITECTURE_BLIP or architecture == ARCHITECTURE_BLIP2:
                image = self._preprocess_image(image_path, model_name)
                tags = self._process_blip(image)
            elif architecture == ARCHITECTURE_VIT:
                tags = self._process_vit(image_path)
            else:
                raise ValueError(f"Invalid architecture: {architecture}")

        # Split the caption into potential tags
        return tags
    def _process_blip(self, image: Image.Image) -> list[str]:
        inputs = self._processor(images=image, return_tensors="pt")

        # generate captions (description of the image)
        outputs = self._model.generate(**inputs)
        caption = self._processor.decode(outputs[0], skip_special_tokens=True)
        tags = [word.lower() for word in caption.split()]
        return tags
    def _process_vit(self, image_path: str) -> list[str]:
        vision_tags = []
        image: Image.Image = Image.open(image_path)


        # code for converting the image and running the model is taken from the link below
        # thanks, SmilingWolf!
        # https://huggingface.co/spaces/SmilingWolf/wd-v1-4-tags/blob/main/app.py

        # convert an image to fit the model
        _, height, _, _ = self._model.get_inputs()[0].shape

        # alpha to white
        image = image.convert('RGBA')
        new_image = Image.new('RGBA', image.size, 'WHITE')
        new_image.paste(image, mask=image)
        image = new_image.convert('RGB')
        image = np.asarray(image)

        # PIL RGB to OpenCV BGR
        image = image[:, :, ::-1]

        image = dbimutils.make_square(image, height)
        image = dbimutils.smart_resize(image, height)
        image = image.astype(np.float32)
        image = np.expand_dims(image, 0)
        
        input_name = self._model.get_inputs()[0].name
        label_name = self._model.get_outputs()[0].name
        confidents = self._model.run([label_name], {input_name: image})[0]

        tags = self._model_tags[:][['name']]
        tags['confidents'] = confidents[0]

        # first 4 items are for rating (general, sensitive, questionable, explicit)
        ratings = dict(tags[:4].values)

        # rest are regular tags
        tags = dict(tags[4:].values)
        threshold = 0.35
        logging.info(f"filtering {len(tags)} tags with threshold {threshold}")
        filtered = {tag: confidence for tag, confidence in tags.items() if confidence > threshold}
        logging.info(f"filtered down to {len(filtered)} tags")
        keys = list(filtered.keys())
        keys = [s.replace("_", " ") for s in keys]
        return keys
    def _preprocess_image(self, image_path: str, model_name: str) -> Image.Image:
        target_size = Interrogator.get_dimensions_for_model(model_name)
        image: Image.Image = Image.open(image_path)
        image.thumbnail(target_size, Image.Resampling.LANCZOS)

        padded_image = Image.new("RGB", target_size, (0, 0, 0))
        padded_image.paste(
            image,
            ((target_size[0] - image.size[0]) // 2, (target_size[1] - image.size[1]) // 2)
        )
        return padded_image

    def _setup_model(self, model_name):
        logging.info(f"Setting up model: {model_name}")
        architecture = Interrogator.get_model_architecture(model_name)
        if architecture == ARCHITECTURE_BLIP:
            self._model = BlipForConditionalGeneration.from_pretrained(model_name)
            self._processor = BlipProcessor.from_pretrained(model_name)
        elif architecture == ARCHITECTURE_BLIP2:
            self._model = Blip2ForConditionalGeneration.from_pretrained(model_name)
            self._processor = Blip2Processor.from_pretrained(model_name)
        elif architecture == ARCHITECTURE_VIT:
            self._setup_wd(model_name)
        else:
            raise ValueError(f"Invalid architecture: {architecture}")
        self._current_model_name = model_name
    def _setup_wd(self, model_name: str):
        model_file = "model.onnx"
        tags_file = "selected_tags.csv"
        repo_id = model_name
        logging.info(f"Downloading wd model: {model_name}")
        model_path = Path(hf_hub_download(
            repo_id=repo_id,
            filename=model_file
        ))
        tags_path = Path(hf_hub_download(
            repo_id=repo_id,
            filename=tags_file,
        ))

        from onnxruntime import InferenceSession
        self._model = InferenceSession(
            str(model_path),
            providers=self._providers,
        )
        logging.info(f"Loaded wd model {model_name} from {model_path}")
        self._model_tags = pd.read_csv(tags_path)


    def _teardown_model(self):
        model_architecture = Interrogator.get_model_architecture(self._current_model_name)
        if model_architecture == "blip" or model_architecture == "blip2":
            logging.info(f"Tearing down model {self._current_model_name}")
            del self._model
            self._model = None
            del self._processor
            self._processor = None
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif model_architecture == ARCHITECTURE_VIT:
            del self._model
            self._model = None
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        else:
            raise RuntimeError(f"not implemented: {model_architecture}")

    @staticmethod
    def get_valid_models() -> list[str]:
        return [
            "SmilingWolf/wd-vit-large-tagger-v3",
        ]

    @staticmethod
    def get_dimensions_for_model(model_name: str):
        dimensions = {
            "Salesforce/blip-image-captioning-base": (224, 224),
            "Salesforce/blip2-opt-2.7b": (224, 224),
            "Salesforce/blip2-flan-t5-xl": (224, 224),
            "SmilingWolf/wd-vit-large-tagger-v3": (256, 256),
        }
        return dimensions[model_name]

    @staticmethod
    def get_model_architecture(model_name: str):
        architecture = {
            "Salesforce/blip-image-captioning-base": ARCHITECTURE_BLIP,
            "Salesforce/blip2-opt-2.7b": ARCHITECTURE_BLIP2,
            "Salesforce/blip2-flan-t5-xl": ARCHITECTURE_BLIP2,
            "SmilingWolf/wd-vit-large-tagger-v3": ARCHITECTURE_VIT,
        }
        return architecture[model_name]
