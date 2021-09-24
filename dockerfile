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
# When switching to mounting the whole project as a volume, it
# seemed wrong to mount it at the existing /home/app directory. So,
# going one level deeper. I think an alternative is to just work from
# /app, but I think some programs like JupyterLab have some issues
# when running from outside the home tree.
#WORKDIR /home/app	
RUN mkdir app && chown $USER_ID app
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
     imagemagick \
	 ffmpeg && \
	 rm -rf /var/lib/apt/lists/*

RUN conda config --add channels conda-forge 
RUN conda install --yes \
	#nodejs'>=12.0.0' \
	matplotlib \
	pandas \
# Doesn't resolve :(. Using Apt-get instead.
#    imagemagick \
	scikit-learn \
	flask \
	jupyterlab  \
    ipykernel>=6 \
    xeus-python \
    ipywidgets && \
	conda clean -ya
RUN conda install --yes -c fastai nbdev 
# Custom repodata a temporary fix. See:
# https://stackoverflow.com/questions/62325068/cannot-install-latest-nodejs-using-conda-on-mac
RUN conda install --yes -c conda-forge nodejs'>=12.0.0' --repodata-fn=repodata.json
RUN conda install -c conda-forge jupyterlab-spellchecker
RUN jupyter labextension install @axlair/jupyterlab_vim
# For jupyter lab 2.x
# RUN jupyter labextension install @ijmbarr/jupyterlab_spellchecker

RUN pip install --upgrade pip
RUN pip install graphviz \
        opencv-python \
        icecream \
        moviepy

# In order to allow the Python package to be edited without
# a rebuild, install all code as a volume. We will still copy the
# files initially, so that things like the below pip install can work.
COPY --chown=$USER_ID ./ ./

# Fix permission issues with ims
# https://stackoverflow.com/a/54230833/754300
RUN rm /etc/ImageMagick-6/policy.xml

# Install our own project as a module.
# This is done so the tests and JupyterLab code can import it.
ENV DISTUTILS_DEBUG=1
RUN pip install -e ./nncolor

# Switching to our new user. Do this at the end, as we need root permissions 
# in order to create folders and install things.
USER app
