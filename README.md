# [sesopenko/interrogate_forever](https://github.com/sesopenko/interrogate_forever)

## Requirements

* Cuda Toolkit
    * Tested with [Cuda Toolkit 12.4](https://developer.nvidia.com/cuda-12-4-0-download-archive)
    * If you're on windows, remember to install cuda toolkit before graphics drivers!
    * Might work with
* [pytorch](https://pytorch.org/get-started/locally/)

It's recommended you configure the HF_HOME env variable to point to somewhere explicit. ie: a docker mount.

## Running locally

Tested on windows 11 with Python 3.12.

```
# this requirements file includes torch for cuda 12.4
pip install -r requirements-windows-cuda124.txt
python main.py watch
```

## Running with Docker

This requires that you have nvidia container toolkit working, which is outside the scope of this guide.

Build image:

```bash
docker build -t sesopenko/interrogate_forever:latest .
```

Run image:

```
docker run --gpus all -it --rm -v "$(pwd):/app" sesopenko/interrogate_forever:latest bash
```

## Running Jobs:

Decide on a unique ID for the new tagging job. UUID is recommended but anything that will never be repeated is
acceptable.

Create a zip file with the following contents:

* an image. Must be jpeg or png
* job.json with the following properties:
    * model_name: `SmilingWolf/wd-vit-large-tagger-v3` (only one supported)
    * job_id: unique ID for your job.
    * input_image_filename: the filename of the image in your zip file

name your zip file <your_unique_id>.zip, ie: `12345.zip`

Copy the zip file to `data/input` and the program will pick up the job within a second. Shortly after it will output a
job_id.json file into data/output. Your tags will be in there.

## License

This software is licenced under GNU GPL V3. The license is included in [LICENSE.txt](LICENSE.txt). If it is missing it
may be read at:

https://www.gnu.org/licenses/gpl-3.0.txt

## Attribution

* [test_assets/attribution.md](test_assets/attribution.md)
*

Copyright © Sean Esopenko 2025