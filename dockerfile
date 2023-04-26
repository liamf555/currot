# FROM python:3.8
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

WORKDIR /app

RUN apt update && apt install -y python3-pip

RUN pip3 install sbx-rl==0.5.0
RUN pip3 install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip3 install wandb
RUN pip3 install Box2d
RUN pip3 install earcut 
RUN pip3 install shimmy
RUN pip3 install tensorboard

COPY ./requirements.txt /app/requirements.txt
RUN pip3 install -r requirements.txt

COPY ./nadaraya-watson /app/nadaraya-watson
RUN cd /app/nadaraya-watson && pip3 install .

COPY ./gym_mxs /app/gym_mxs
RUN cd /app/gym_mxs && pip3 install -e .


COPY . .

# COPY ./analysis_scripts /app/analysis_scripts
# COPY /home/tu18537/dev/mxs/pymxs/gym_mxs /app/gym_mxs
# COPY /home/tu18537/dev/mxs/pymxs/inertia /app/inertia
# COPY /home/tu18537/dev/mxs/pymxs/models /app/models
# COPY /home/tu18537/dev/mxs/pymxs/processing_scripts /app/processing_scripts
# COPY /home/tu18537/dev/mxs/pymxs/pyaerso /app/pyaerso
# COPY ./pymxs_sbx_run.py /app/pymxs_sbx_run.py
# COPY ./pymxs_sbx_box.py /app/pymxs_sbx_box.py

ENV WANDB_API_KEY="ea17412f95c94dfcc41410f554ef62a1aff388ab"

ENTRYPOINT ["python3", "run.py"]
CMD ["--type", "alp_gmm", "--learner", "ppo", "--env", "mxs_box2d"]
