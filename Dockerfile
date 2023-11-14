FROM continuumio/miniconda3 AS build

RUN apt-get update

# Installing git and SSH in case required by conda
RUN apt-get install -y git openssh-client openssh-server ca-certificates
RUN git config --global http.sslVerify false

# Authorize SSH Host
RUN mkdir -p /root/.ssh
RUN chmod 0700 /root/.ssh
RUN ssh-keyscan github.com > /root/.ssh/known_hosts

# Updating conda
RUN conda update -n base -c defaults conda

WORKDIR /app
COPY environment.yml .

# Creating the environment
RUN --mount=type=ssh,id=github_ssh_key conda env create -f environment.yml

# Installing conda-pack
RUN conda install -c conda-forge conda-pack

# Use conda-pack to create a standalone environment in /venv:
RUN conda-pack -n my_env_name -o /tmp/env.tar && \
mkdir /venv && cd /venv && tar xf /tmp/env.tar && \
rm /tmp/env.tar

# We've put venv in the same path it'll be in the final image,
# so now fix up paths:
RUN /venv/bin/conda-unpack

# RUN conda clean -a -y

# The runtime-stage image
FROM debian:buster AS runtime
LABEL version="1.0.0"
LABEL maintainer="Agnès Regaud"
LABEL vcs-url="https://github.com/agnesR23/Projet7_OC_Datascientist"
LABEL description="Streamlit-Flask App"

# Copy /venv from the previous stage
COPY --from=build /venv /venv

# Copy /app from the previous stage
COPY --from=build /app /app

WORKDIR /app

EXPOSE 8501

# Run the code with the environment activated
SHELL ["/bin/bash", "-c"]

# change file permission to prevent access denied error
RUN chmod +x /app/bash/docker_run.bash
ENTRYPOINT ["/app/bash/docker_run.bash"]







