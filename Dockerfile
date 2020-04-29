FROM python:3.6.3

LABEL maintainer Dan Napierski (ISI) <dan.napierski@toptal.com>

# Create app directory
WORKDIR /aida/src/

# Update
RUN apt-get update && apt-get install -y apt-utils wget tree nano git libgl1-mesa-glx
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-4.5.1-Linux-x86_64.sh
RUN chmod +x ./Miniconda3-4.5.1-Linux-x86_64.sh
RUN ./Miniconda3-4.5.1-Linux-x86_64.sh -b -p ~/conda
ENV PATH="/root/conda/bin:${PATH}"
RUN conda update -n base -c defaults conda && conda -V && conda install setuptools

WORKDIR /home/bobby/aida_copy/AIDA/
RUN git clone https://github.com/NextCenturyCorporation/AIDA-Interchange-Format.git
RUN ls -al /home/bobby/aida_copy/AIDA/
ENV PYTHONPATH "/aida/src/:/aida/src/src/:/home/bobby/aida_copy/AIDA/AIDA-Interchange-Format/python/"

WORKDIR /aida/src/
COPY aida-env.txt ./
RUN conda create --name aida-env --file ./aida-env.txt -c conda-forge tensorflow-gpu=1.15 rdflib=4.2.2 
RUN echo "source activate aida-env" >> ~/.bashrc

# Bundle app source
COPY . .
# Open port
EXPOSE 8082

LABEL name="AIDA Grounding and Merging"
LABEL version=0
LABEL revision=1

CMD [ "/bin/bash", "" ]
