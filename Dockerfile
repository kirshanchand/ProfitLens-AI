FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

ENV PROFITLENS_PROJECT_ROOT=/app
ENV STREAMLIT_SERVER_HEADLESS=true

EXPOSE 7860

CMD ["streamlit", "run", "streamlit_app/app.py", "--server.address=0.0.0.0", "--server.port=7860"]
