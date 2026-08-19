from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    'owner': 'admin',
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

with DAG(
    dag_id='pokretanje_3_fajla_pipeline',
    default_args=default_args,
    description='Pokretanje 3 skripte redom jedna za drugom',
    schedule_interval='@daily',          
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:

    
    pokreni_fajl_1 = BashOperator(
        task_id='skripta_1_skupljanje_podataka',
        bash_command='python /opt/airflow/utils/fajl1.py'
    )


    pokreni_fajl_2 = BashOperator(
        task_id='skripta_2_obrada_podataka',
        bash_command='python /opt/airflow/utils/fajl2.py'
    )


    pokreni_fajl_3 = BashOperator(
        task_id='skripta_3_treniranje_modela',
        bash_command='python /opt/airflow/utils/fajl3.py'
    )


    pokreni_fajl_1 >> pokreni_fajl_2 >> pokreni_fajl_3