import json
import pathlib
import airflow
import Solar_prediction_functions as solar
import requests
import requests.exceptions as requests_exceptions
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from xgboost import XGBRegressor
from datetime import datetime

dag = DAG(
 dag_id="Dataset_management",
 start_date=airflow.utils.dates.days_ago(0),
)

get_dataset_energy_production = PythonOperator(
    task_id='get_dataset_energy_production',
    python_callable=solar.get_dataset,
    op_kwargs={"dataset_path": "Solar_Energy_Production.csv","values": ['name', 'id', 'address', 'date', 'kWh']},
    dag=dag,
)

get_dataset_sites = PythonOperator(
    task_id='get_dataset_sites',
    python_callable=solar.get_dataset,
    op_kwargs={"dataset_path": "Solar_Photovoltaic_Sites.csv","values": ['id', 'latitude', 'longitude']},
    dag=dag,
)

combine_datasets = PythonOperator(
    task_id='combine_datasets',
    python_callable=solar.combine_datasets,
    op_kwargs={'_dataset1': 'get_dataset_energy_production','_dataset2': 'get_dataset_sites','on': 'id','how': 'left'},
    dag=dag,
)

preprocessing = PythonOperator(
    task_id='preprocessing',
    python_callable=solar.preprocessing,
    op_kwargs={'data': 'combine_datasets'},
    dag=dag,
)

get_api_data = PythonOperator(
    task_id='get_api_data',
    python_callable=solar.get_api_data,
    op_kwargs={'data': 'preprocessing'},
    dag=dag,
)

process_api_data = PythonOperator(
    task_id='process_api_data',
    python_callable=solar.process_api_data,
    op_kwargs={'data': 'get_api_data'},
    dag=dag,
)

save_dataset = PythonOperator(
    task_id='save_dataset',
    python_callable=solar.save_dataset,
    op_kwargs={'dataset': 'process_api_data'},
    dag=dag,
)

model_training = PythonOperator(
    task_id='model_training',
    python_callable=solar.model_training,
    op_kwargs={'data': 'process_api_data'},
    dag=dag,
)

make_prediction = PythonOperator(
    task_id='make_prediction',
    python_callable=solar.make_prediction,
    op_kwargs={'x': '42.3601','y':'-71.0589','model_name': 'model.pkl', 'data': 'get_dataset'},
    dag=dag,
)


get_dataset_energy_production >> get_dataset_sites >> combine_datasets >> preprocessing >> get_api_data >> process_api_data >> save_dataset >> model_training  >> make_prediction