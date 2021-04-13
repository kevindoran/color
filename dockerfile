FROM pytorch/pytorch

# Every thing seems to work a lot easier if we explicitly create a non-root 
# user. This follows the advice from: https://vsupalov.com/docker-shared-permissions/
# Creating the user made running Jupiter Lab work.
ARG USER_ID=1001
ARG GROUP_ID=101
RUN addgroup -gid $GROUP_ID app
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID app

# Update: even better, create a user, and let the home dir be created as a result.
# Recommended way to create a subdirectory is to not rely on WORKDIR to create
# it. WORKDIR doesn't have any chown capabilities, and we need the app
# directory to be user owned, as pytest creates some cache files.
# https://github.com/moby/moby/issues/36677
#RUN mkdir app && chown 1001 app
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

RUN conda config --add channels conda-forge && \
	conda install --yes \
	nodejs'>=12.0.0' \
	matplotlib \
	pandas \
	scikit-learn \
# Leads to conflicts.
# python-graphviz \
	flask \
	jupyterlab && \
	conda clean -ya
RUN jupyter labextension install @axlair/jupyterlab_vim

RUN pip install graphviz

COPY --chown=$USER_ID ./ ./
# The root user is needed to install things and create folders. Switching back
# to the app user, which is the user that will run any container commands.
USER app

# Install our own project as a module.
# This is done so the tests can import it.
# RUN pip install .
