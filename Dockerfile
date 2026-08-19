FROM apache/airflow:2.9.2-python3.12

USER root

RUN mkdir -p /opt/airflow/models /opt/airflow/results /opt/airflow/utils /opt/airflow/streamlit_app && \
    chown -R airflow:root /opt/airflow

USER airflow

COPY --chown=airflow:root requirements.txt /opt/airflow/requirements.txt

RUN pip install --no-cache-dir -r /opt/airflow/requirements.txt

COPY --chown=airflow:root . /opt/airflow