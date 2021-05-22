FROM pytorch/pytorch

# Every thing seems to work a lot easier if we explicitly create a non-root 
# user. This follows the advice from: https://vsupalov.com/docker-shared-permissions/
# Creating the user made running Jupiter Lab work.
# Creating a new (non-root) user makes a lot of things easier:
# - there will be a home directory with user permissions
# - no need to manually create a directory to work from
# - avoid the "no username" being displayed when using iterative mode.
ARG USER_ID=1001
ARG GROUP_ID=101
RUN addgroup --gid $GROUP_ID app
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID app
WORKDIR /app

# These next two folders will be where we will mount our local data and out
# directories. We create them manually (automatic creation when mounting will
# create them as being owned by root, and then our program cannot use them).
RUN mkdir data && chown $USER_ID data
RUN mkdir out && chown $USER_ID out

RUN apt-get update && apt-get install -y --no-install-recommends \
	 libsm6 \
	 libxext6 \
     libxrender-dev \
	 ffmpeg && \
	 rm -rf /var/lib/apt/lists/*

RUN conda config --add channels conda-forge 
RUN conda install --yes \
	#nodejs'>=12.0.0' \
	matplotlib \
	pandas \
	scikit-learn \
	flask \
	jupyterlab  \
    ipywidgets && \
	conda clean -ya
RUN conda install --yes -c fastai nbdev 
# Custom repodata a temporary fix. See:
# https://stackoverflow.com/questions/62325068/cannot-install-latest-nodejs-using-conda-on-mac
RUN conda install --yes -c conda-forge nodejs'>=12.0.0' --repodata-fn=repodata.json
RUN jupyter labextension install @axlair/jupyterlab_vim

RUN pip install graphviz

COPY --chown=$USER_ID ./ ./

# Switching to our new user. Do this at the end, as we need root permissions 
# in order to create folders and install things.
USER app

# Install our own project as a module.
# This is done so the tests can import it.
# RUN pip install .
