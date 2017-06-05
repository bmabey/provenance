FROM andrewosh/binder-base

MAINTAINER Ben Mabey <ben@benmabey.com>

USER root

RUN apt-get update -y && \
    apt-get install -y postgresql postgresql-contrib && \
    service postgresql start

USER main

ADD environment.yml /home/main/environment.yml
RUN /home/main/anaconda/bin/conda install nb_conda_kernels && \
    cd /home/main &&  /home/main/anaconda/bin/conda env create && \
    /bin/bash -c "source /home/main/anaconda/bin/activate provenance-dev && pip install git+https://github.com/bmabey/provenance"


CMD ["/bin/bash"]
