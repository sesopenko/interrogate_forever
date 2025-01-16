import logging

from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from typing import Optional
import threading

logger = logging.getLogger(__name__)


class Interrogator:
    def __init__(self):
        self._current_model_name: Optional[str] = None
        self._current_model: Optional[Blip2ForConditionalGeneration] = None
        self._processor: Optional[Blip2Processor] = None
        self._mutex = threading.Lock()

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
            image = self._preprocess_image(image_path, model_name)

            # prepare inputs for the model
            inputs = self._processor(images=image, return_tensors="pt")

            # generate captions (description of the image)
            outputs = self._model.generate(**inputs)
            caption = self._processor.decode(outputs[0], skip_special_tokens=True)

        # Split the caption into potential tags
        tags = [word.lower() for word in caption.split()]
        return tags

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
        self._current_model_name = model_name
        self._model = Blip2ForConditionalGeneration.from_pretrained(model_name)
        self._processor = Blip2Processor.from_pretrained(model_name)

    def _teardown_model(self):
        logging.info(f"Tearing down model {self._current_model_name}")
        del self._model
        self._model = None
        del self._processor
        self._processor = None

    @staticmethod
    def get_valid_models() -> list[str]:
        return [
            "Salesforce/blip-image-captioning-base",
            "Salesforce/blip2-opt-2.7b",
            "Salesforce/blip2-flan-t5-xl",
        ]

    @staticmethod
    def get_dimensions_for_model(model_name: str):
        dimensions = {
            "Salesforce/blip-image-captioning-base": (224, 224),
            "Salesforce/blip2-opt-2.7b": (224, 224),
            "Salesforce/blip2-flan-t5-xl": (224, 224),
        }
        return dimensions[model_name]
