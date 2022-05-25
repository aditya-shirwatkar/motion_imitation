# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.224.2/containers/ubuntu/.devcontainer/base.Dockerfile

# [Choice] Ubuntu version (use hirsuite or bionic on local arm64/Apple Silicon): hirsute, focal, bionic
ARG VARIANT="focal"
FROM mcr.microsoft.com/vscode/devcontainers/base:0-${VARIANT}

RUN apt-get update && apt-get upgrade -y

# [For Rendering]
RUN apt-get install libxext6 libxrender1 libxtst6 libxi6 libxxf86vm1 mesa-utils python3-pip -y

# For motion imitation
# RUN virtualenv --python=python3.6 motion_imitation
RUN ln -s /usr/bin/python3 /usr/bin/python

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>


# # [Optional] Allow the vscode user to pip install globally w/o sudo
# ENV PIP_TARGET=/usr/local/pip-global
# ENV PYTHONPATH=${PIP_TARGET}:${PYTHONPATH}
# ENV PATH=${PIP_TARGET}/bin:${PATH}
# RUN mkdir -p ${PIP_TARGET} \
#     && chown vscode:root ${PIP_TARGET} \
#     && echo "if [ \"\$(stat -c '%U' ${PIP_TARGET})\" != \"vscode\" ]; then chown -R vscode:root ${PIP_TARGET}; fi" \
#         | tee -a /root/.bashrc /home/vscode/.bashrc /root/.zshrc >> /home/vscode/.zshrc 

RUN python -m pip install --user --upgrade pip

# # [Optional] If your pip requirements rarely change, uncomment this section to add them to the image.
COPY .devcontainer/sync_deps.py /workspaces/session/post_start
# copy requirements list into layer of dev image
COPY requirements.txt /workspaces/session/

RUN pip install -r /workspaces/session/requirements.txt
RUN python3 -m pip install --user crocoddyl

# RUN python setup.py install --user
# RUN pip install -e .
