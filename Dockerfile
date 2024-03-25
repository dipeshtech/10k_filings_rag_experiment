FROM python:3.11-slim-bullseye
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app
COPY --chown=user . $HOME/app
COPY ./requirements.txt ~/app/requirements.txt
RUN pip config --user set global.progress_bar off
RUN pip install -r requirements.txt --progress-bar off
COPY . .
CMD ["streamlit", "run", "app.py"]
