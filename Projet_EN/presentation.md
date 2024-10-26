
 Project Title: Solar Radiation Prediction Using Machine Learning

 2: Project Overview

    Objective:  to build a machine learning model to predict solar radiation accurately, considering weather variability across different regions.
    Importance: Briefly discuss why solar radiation prediction is essential (e.g., renewable energy planning, grid stability, environmental impact).

 3: Data Sources

    List data sources:
        Calgary’s Solar Energy Production Data
        Solar Photovoltaic Sites Data from Calgary
        Additional Weather Data API (like Open Meteo)
    dataset’s key details, target features like solar radiation (watts per square meter) and relevant weather attributes like temperature, humidity, and wind speed​

 4: Data Preparation and Feature Engineering

    Merging Datasets: Calgary solar energy data and site information were merged based on location.
    Cleaning and Preprocessing: List key steps, handling missing values and scaling numerical features.
    Feature Engineering: Describe any calculated features, sunrise/sunset times and daily averages.

 5: Model Selection and Training

    Chosen Model: XGBoost Regressor.
    Why XGBoost?: its handling of structured data and performance.
    Hyperparameter Tuning: Mention if you used cross-validation or grid search to optimize parameters.

 6: Testing with New Dataset

    Testing Dataset: testing (e.g., from European Data Portal or EDF).
    Evaluation Metrics: the metrics used to assess performance, like Mean Absolute Error (MAE) and R-squared score.
    Results: Include initial findings on model accuracy with the new data, mentioning any observed generalization issues or model adjustments made.

 7: Pipeline and Deployment with Airflow

    Airflow Pipeline: Outline the main steps automated in the pipeline:
        Data Collection
        Preprocessing
        Model Training
        Evaluation
    Technical Setup: using Docker for local Airflow installation and designing DAGs for task automation.

