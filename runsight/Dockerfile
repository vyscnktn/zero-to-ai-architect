# AWS Lambda icin resmi Python 3.13 base image (.venv'deki Python surumuyle ayni)
FROM public.ecr.aws/lambda/python:3.13

# TensorFlow bazi runtime kutuphaneleri (libgomp) icin gerekli
RUN dnf install -y libgomp && dnf clean all

# Once sadece bagimliliklari kopyala: kod degisince (8_gru_deploy.py) Docker
# katman onbellegi sayesinde tum kutuphaneler yeniden kurulmaz.
COPY requirements-deploy.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements-deploy.txt

# Model, on isleme nesneleri (scaler'lar) ve handler kodu
COPY gru_model.keras ${LAMBDA_TASK_ROOT}/
COPY gru_preprocessing.pkl ${LAMBDA_TASK_ROOT}/
COPY 8_gru_deploy.py ${LAMBDA_TASK_ROOT}/

CMD ["8_gru_deploy.lambda_handler"]
