# 
FROM python:3.9

# 
WORKDIR /code

# 
COPY ./requirements.txt /code/requirements.txt

# 
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 
COPY ./src /code/src

# 
CMD ["fastapi", "run", "src/fastapi.py", "--port", "80"]