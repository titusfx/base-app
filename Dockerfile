# Use an official Python runtime as a parent image
FROM python:3.10.12-slim

# Set environment variables
ENV POETRY_VERSION=1.8.3
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

RUN pip install "poetry==$POETRY_VERSION"


# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY pyproject.toml poetry.lock* /app/

# Install the dependencies using Poetry
RUN poetry config virtualenvs.create false && poetry install --no-root --no-dev

COPY . /app

RUN ls -a

# Make port 7860 available to the world outside this container
EXPOSE $PORT
