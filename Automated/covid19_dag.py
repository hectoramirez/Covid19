from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
import datetime as dt

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': dt.datetime(2020, 4, 1, 5, 00, 00),
    'retries': 1
}

dag = DAG('covid',
          default_args=default_args,
          catchup=False,
          schedule_interval='0 5 * * *'
          )

t1 = BashOperator(
    task_id='covidPlots',
    bash_command='python3.7 ~/GitHub/Covid19/Automated/covid_func.py ',
    dag=dag)


t2 = BashOperator(
    task_id='gitPush',
    bash_command='python3.7 ~/GitHub/Covid19/Automated/git_push.py ',
    dag=dag)

t3 = BashOperator(
    task_id='gitPush_repit',
    bash_command='python3.7 ~/GitHub/Covid19/Automated/git_push.py ',
    dag=dag)

t1 >> t2 >> t3
