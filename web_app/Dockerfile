# pull official base image
FROM python:3.8.5

# set work directory
WORKDIR /usr/src/app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install dependencies
RUN pip install --upgrade pip
RUN pip install  --upgrade setuptools
COPY ./requirements.txt .
RUN pip install  --no-cache-dir --ignore-installed -r requirements.txt 

# copy project
COPY . .