FROM python:3.13

RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get install -y ffmpeg

RUN pip install poetry==2.1.3

# Configuring poetry
RUN poetry config virtualenvs.create false

# Copying requirements of a project
COPY pyproject.toml poetry.lock /app/
WORKDIR /app
COPY . /app/

# Installing requirements
RUN poetry install

ENV PYTHONPATH "${PYTHONPATH}:."

CMD ["/usr/local/bin/python", "-m", "src.main"]
