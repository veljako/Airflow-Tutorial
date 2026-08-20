import sys
import os

sys.path.append('/opt/airflow')

from datetime import datetime
from airflow.decorators import dag
from utils.fun import preprocesiranje, treniranje, evaluacija

# Definišemo DAG (Directed Acyclic Graph)
# koji Airflow izvrsava po zadatom rasporedu
@dag(
    dag_id='moj_pipeline', # jedinstveno ime pipeline-a u Airflow-u
    schedule='@daily',     # pipeline se pokreće jednom dnevno
    start_date=datetime(2024, 1, 1), # datum od kog Airflow počinje da računa pokretanja
    catchup=False, # False = ne pokreći retroaktivno propuštena izvršavanja 
                   # (npr. ako je DAG kreiran kasnije, neće "nadoknađivati" prošle dane)
)
def moj_pipeline():

    # Definišemo redosled izvršavanja zadataka (task dependency).
    # ">>" znači "posle ovoga ide sledeće" - tj. izvršava se sekvencijalno:
    # prvo preprocesiranje, pa treniranje, pa na kraju evaluacija
    preprocesiranje() >> treniranje() >> evaluacija()

# Pozivamo funkciju da bi Airflow registrovao DAG
moj_pipeline()