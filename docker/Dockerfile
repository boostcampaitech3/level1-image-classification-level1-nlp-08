FROM python:3.8.8-slim-buster

COPY ./ ./

RUN apt-get update && apt-get upgrade -y && apt-get install -y git

RUN pip install mlflow>=1.0 \
    && pip install numpy \
    && pip install pandas \
    && pip install scikit-learn 
    
RUN mlflow run docker-example --experiment-name my-first-experiment --no-conda

EXPOSE 5001

CMD ["mlflow", "ui", "--host", "0.0.0.0:5001"]