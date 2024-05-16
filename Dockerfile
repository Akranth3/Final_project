# 
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

# 
WORKDIR /code

# 
COPY ./requirements.txt /code/requirements.txt

# 
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 
COPY ./src /code/src
COPY ./utils /code/utils
COPY ./weights /code/weights

# 
CMD ["fastapi", "run", "src/fast_api.py", "--port", "8080"]