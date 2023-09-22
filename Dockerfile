FROM python:3.11-slim-buster 

RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get install -y ffmpeg

RUN pip install poetry==1.4.2

# Configuring poetry
RUN poetry config virtualenvs.create false

# Copying requirements of a project
COPY pyproject.toml poetry.lock /app/
WORKDIR /app

# Installing requirements
RUN poetry install --only main

# Copying actual application
COPY . /app/
RUN poetry install 

ENV PYTHONPATH "${PYTHONPATH}:."

CMD ["/usr/local/bin/python", "-m", "src.main"]