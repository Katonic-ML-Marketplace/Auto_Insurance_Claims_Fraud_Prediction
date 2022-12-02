FROM python:3.8.2-slim

RUN mkdir -p data model image

COPY app.py .
COPY data/insurance_fraud_claims.csv data/.
COPY data/data_req.csv data/.
COPY model/final_model.sav model/.
COPY image/logo.png image/.
COPY image/favicon.ico image/.
COPY requirements.txt .

RUN pip install -r requirements.txt

CMD streamlit run app.py --server.port=8050 --server.address=0.0.0.0 --logger.level error --server.fileWatcherType=none
