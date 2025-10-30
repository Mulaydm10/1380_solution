ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:25.01-py3
FROM ${FROM_IMAGE_NAME}

ENV PYTHONPATH /workspace/yacup_scene_gen
WORKDIR /workspace/yacup_scene_gen

ADD requirements.txt .
RUN pip install -r requirements.txt

COPY . .

ENTRYPOINT ["python3", "inference.py", "--data_path", "input_data", "--save_dir", "output_data", "--ckpt_path", "checkpoints/ckpt/ema.pt", "--config_path", "config.py"]