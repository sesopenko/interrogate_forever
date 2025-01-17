# sesopenko/interrogate_forever

## Requirements

* Cuda Toolkit
  * Tested with [Cuda Toolkit 12.4](https://developer.nvidia.com/cuda-12-4-0-download-archive) 
  * If you're on windows, remember to install cuda toolkit before graphics drivers!
  * Might work with 
* [pytorch](https://pytorch.org/get-started/locally/)

It's recommended you configure the HF_HOME env variable to point to somewhere explicit. ie: a docker mount.

## Docker

Build image:

```bash
docker build -t sesopenko/interrogate_forever:latest .
```
Run image:
```
docker run --gpus all -it --rm -v "$(pwd):/app" sesopenko/interrogate_forever:latest bash
```