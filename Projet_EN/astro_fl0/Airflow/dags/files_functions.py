import os
import pandas as pd
import io
def get_files_directory():
    current_directory = os.getcwd() + "/files/"
    print(f"Les données sont stockées : {current_directory}")
    return str(current_directory)

def get_dataset(dataset_path, values):
    CSV_path = get_files_directory() + dataset_path
    df = pd.read_csv(CSV_path)
    df = df[values]
    print(df)
    print(df.keys())
    return df.to_json()

def combine_datasets(_dataset1, _dataset2, on, how, **context):
    dataset1 = pd.read_json(io.StringIO(context["ti"].xcom_pull(task_ids=_dataset1)))
    dataset2 = pd.read_json(io.StringIO(context["ti"].xcom_pull(task_ids=_dataset2)))
    merged_dataset = pd.merge(dataset1, dataset2, on=on, how=how)
    print(merged_dataset.head(5))
    return merged_dataset.to_json()

def preprocessing(data, **context):
    dataset = pd.read_json(io.StringIO(context["ti"].xcom_pull(task_ids=data)))
    dataset["date"] = pd.to_datetime(dataset["date"])
    dataset.rename(columns={"kWh": "GHI"}, inplace=True)
    dataset = dataset.sort_values(by=["name", "date"])
    print(type(dataset["id"]))
    return dataset.to_json()

def save_dataset(dataset, **context):
    df = pd.read_json(io.StringIO(context["ti"].xcom_pull(task_ids=dataset)))
    print(df.keys())
    df.to_csv(get_files_directory() + "solar_prediction.csv", index=False)
    print("Dataset saved")
    return 0