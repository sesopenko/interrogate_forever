FROM nvcr.io/nvidia/pytorch:24.12-py3

# Set up environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Set the working directory
WORKDIR /app

COPY requirements-docker-cuda124.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-docker-cuda124.txt

COPY ./cli /app/cli
COPY ./core /app/core
COPY ./test_assets /app/test_assets
COPY main.py /app/main.py

# distribute license
COPY LICENSE.txt /app/LICENSE.txt

# make it a little easier for people to reverse engineer
COPY README.md /app/README.md
COPY Dockerfile /app/Dockerfile

ENTRYPOINT ["python", "main.py", "watch"]