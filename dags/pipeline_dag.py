import sys
import os

sys.path.append('/opt/airflow')

from datetime import datetime
from airflow.decorators import dag
from utils.fun import preprocesiranje, treniranje, evaluacija

@dag(
    dag_id='moj_pipeline',
    schedule='@daily',
    start_date=datetime(2024, 1, 1),
    catchup=False,
)
def moj_pipeline():
    preprocesiranje() >> treniranje() >> evaluacija()

moj_pipeline()