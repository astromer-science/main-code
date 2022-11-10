FROM tensorflow/tensorflow:2.8.0-gpu
# ==== USER CREATION ===
ARG USER_ID
ARG GROUP_ID
RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
# ==== UPDATE SISTEM ====
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-get update
RUN apt install -y graphviz
RUN apt install nano htop wget
# ==== PIP SETUP ====
RUN python -m pip install --upgrade pip
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install dvipng texlive-latex-extra texlive-fonts-recommended cm-super
# ==== BUILDING WORKING DIR ====
USER user
WORKDIR /home/
ADD ./requirements.txt ./requirements.txt
# ==== INSTALL PYTHON REQUIREMENTS ====
RUN pip install -r requirements.txt --no-cache-dir
ENV PATH=$PATH:~/.local/bin
# ==== EXPOSE PORTS ====
EXPOSE 8888
EXPOSE 6006
