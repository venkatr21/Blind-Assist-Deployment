FROM python:3.7
WORKDIR /app
ADD . /app
RUN pip install -r requirements.txt
EXPOSE 5000
ENV NAME vision
CMD ["python", "app.py"]

