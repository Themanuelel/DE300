from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import boto3
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import requests
from scrapy import Selector
import os
import re

pip install scrapy

# Define default_args for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'eda_workflow',
    default_args=default_args,
    description='An EDA workflow with sklearn and Spark',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2023, 1, 1),
    catchup=False,
)

# Function to load data from S3
def load_data_from_s3(**kwargs):
    s3 = boto3.client('s3')
    bucket_name = 'de300spring2024-eman'
    key = 'heart_disease.csv'
    local_path = '/tmp/heart_disease.csv'
    s3.download_file(bucket_name, key, local_path)
    return local_path

# EDA Function
def eda_preprocessing(**kwargs):
    df = pd.read_csv("/tmp/heart_disease.csv")
    df = df[0:899]

    columns_to_keep = ['age', 'sex', 'painloc', 'painexer', 'cp', 'trestbps', 
                       'smoke', 'fbs', 'prop', 'nitr', 'pro', 'diuretic', 
                       'thaldur', 'thalach', 'exang', 'oldpeak', 'slope', 'target']
    df_subset = df[columns_to_keep]

    # Fill missing values
    df_subset['painloc'] = df_subset['painloc'].fillna(df_subset['painloc'].mode()[0])
    df_subset['painexer'] = df_subset['painexer'].fillna(df_subset['painexer'].mode()[0])
    df_subset.loc[df_subset['trestbps'] < 100, 'trestbps'] = 100
    df_subset.loc[df_subset['oldpeak'] < 0, 'oldpeak'] = 0
    df_subset.loc[df_subset['oldpeak'] > 4, 'oldpeak'] = 4
    mean_thaldur = round(df_subset['thaldur'].mean(), 1)
    mean_thalach = round(df_subset['thalach'].mean(), 1)
    df_subset['thaldur'].fillna(mean_thaldur, inplace=True)
    df_subset['thalach'].fillna(mean_thalach, inplace=True)
    mode_fbs = df_subset['fbs'].mode()[0]
    mode_prop = df_subset['prop'].mode()[0]
    mode_nitr = df_subset['nitr'].mode()[0]
    mode_pro = df_subset['pro'].mode()[0]
    mode_diuretic = df_subset['diuretic'].mode()[0]
    mode_exang = df_subset['exang'].mode()[0]
    mode_slope = df_subset['slope'].mode()[0]
    df_subset['fbs'].fillna(mode_fbs, inplace=True)
    df_subset['prop'].fillna(mode_prop, inplace=True)
    df_subset['nitr'].fillna(mode_nitr, inplace=True)
    df_subset['pro'].fillna(mode_pro, inplace=True)
    df_subset['diuretic'].fillna(mode_diuretic, inplace=True)
    df_subset['exang'].fillna(mode_exang, inplace=True)
    df_subset['slope'].fillna(mode_slope, inplace=True)
    df_subset.loc[df_subset['fbs'] > 1, 'fbs'] = mode_fbs
    df_subset.loc[df_subset['prop'] > 1, 'prop'] = mode_prop
    df_subset.loc[df_subset['nitr'] > 1, 'nitr'] = mode_nitr
    df_subset.loc[df_subset['pro'] > 1, 'pro'] = mode_pro
    df_subset.loc[df_subset['diuretic'] > 1, 'diuretic'] = mode_diuretic
    subs_cols = ['trestbps', 'oldpeak', 'thaldur', 'thalach']
    df_subs = df_subset[subs_cols]
    skewness = df_subs.skew()
    for col in df_subs.columns:
        if abs(skewness[col]) < 0.5:
            df_subs[col].fillna(round(df_subs[col].mean(), 1), inplace=True)
        else:
            df_subs[col].fillna(df_subs[col].median(), inplace=True)
    df_subset[subs_cols] = df_subs

    # Impute smoke column using source 1
    def get_smoking_percentage_src1(age):
        if 15 <= age <= 17:
            return .016
        elif 18 <= age <= 24:
            return .073
        elif 25 <= age <= 34:
            return .109
        elif 35 <= age <= 44:
            return .109
        elif 45 <= age <= 54:
            return .138
        elif 55 <= age <= 64:
            return .149
        elif 65 <= age <= 74:
            return .087
        elif age >= 75:
            return .029
        else:
            return None
    df_subset['smoking_src1'] = df_subset.apply(lambda row: row['smoke'] if row['smoke'] in [0, 1] else get_smoking_percentage_src1(row['age']), axis=1)

    # Impute smoke column using source 2
    def get_smoking_percentage_src2(age, sex):
        if sex == 0:  # Female
            if 18 <= age <= 24:
                return .053
            elif 25 <= age <= 44:
                return .126
            elif 45 <= age <= 64:
                return .149
            elif age >= 65:
                return .083
        elif sex == 1:
            if 18 <= age <= 24:
                return round(.053 * (.131 / .101), 3)
            elif 25 <= age <= 44:
                return round(.126 * (.131 / .101), 3)
            elif 45 <= age <= 64:
                return round(.149 * (.131 / .101), 3)
            elif age >= 65:
                return round(.083 * (.131 / .101), 3)
        return None
    df_subset['smoke_src2'] = df_subset.apply(lambda row: row['smoke'] if row['smoke'] in [0, 1] else get_smoking_percentage_src2(row['age'], row['sex']), axis=1)
    
    # Drop the original 'smoke' column
    df_subset.drop(columns=['smoke'], inplace=True)
    
    # Save the modified DataFrame
    df_subset.to_csv("/tmp/heart_disease_subset.csv", index=False)
    return "/tmp/heart_disease_subset.csv"

# Function for feature engineering strategy 1
def feature_engineering_1(file_path, **kwargs):
    df = pd.read_csv(file_path)
    df['feature1'] = df['original_feature'] ** 2
    fe_path = file_path.replace('.csv', '_fe1.csv')
    df.to_csv(fe_path, index=False)
    return fe_path

# Function for feature engineering strategy 2
def feature_engineering_2(file_path, **kwargs):
    df = pd.read_csv(file_path)
    df['feature2'] = df['original_feature'] ** 0.5
    fe_path = file_path.replace('.csv', '_fe2.csv')
    df.to_csv(fe_path, index=False)
    return fe_path

# Model training function
def train_models(file_path, **kwargs):
    df = pd.read_csv(file_path)
    X = df.drop(columns=['target'])
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
    
    logistic_model = LogisticRegression(max_iter=1000)
    random_forest_model = RandomForestClassifier(max_depth=100)
    svm_model = SVC(C=1000)
    
    logistic_model.fit(X_train, y_train)
    random_forest_model.fit(X_train, y_train)
    svm_model.fit(X_train, y_train)
    
    logistic_pred = logistic_model.predict(X_test)
    random_forest_pred = random_forest_model.predict(X_test)
    svm_pred = svm_model.predict(X_test)
    
    logistic_accuracy = accuracy_score(y_test, logistic_pred)
    random_forest_accuracy = accuracy_score(y_test, random_forest_pred)
    svm_accuracy = accuracy_score(y_test, svm_pred)
    
    logistic_report = classification_report(y_test, logistic_pred)
    random_forest_report = classification_report(y_test, random_forest_pred)
    svm_report = classification_report(y_test, svm_pred)
    
    logistic_cv_scores = cross_val_score(logistic_model, X_train, y_train, cv=5)
    random_forest_cv_scores = cross_val_score(random_forest_model, X_train, y_train, cv=5)
    svm_cv_scores = cross_val_score(svm_model, X_train, y_train, cv=5)
    
    logistic_mean_cv_score = logistic_cv_scores.mean()
    random_forest_mean_cv_score = random_forest_cv_scores.mean()
    svm_mean_cv_score = svm_cv_scores.mean()
    
    print("Logistic Regression Model Accuracy:", logistic_accuracy)
    print("Logistic Regression Model Classification Report:\n", logistic_report)
    print("Random Forest Model Accuracy:", random_forest_accuracy)
    print("Random Forest Model Classification Report:\n", random_forest_report)
    print("SVM Model Accuracy:", svm_accuracy)
    print("SVM Model Classification Report:\n", svm_report)
    print("Logistic Regression Model Mean Cross-Validation Score:", logistic_mean_cv_score)
    print("Random Forest Model Mean Cross-Validation Score:", random_forest_mean_cv_score)
    print("SVM Model Mean Cross-Validation Score:", svm_mean_cv_score)

# Define Airflow tasks
load_data_task = PythonOperator(
    task_id='load_data_from_s3',
    python_callable=load_data_from_s3,
    dag=dag,
)

eda_task = PythonOperator(
    task_id='eda_preprocessing',
    python_callable=eda_preprocessing,
    provide_context=True,
    dag=dag,
)

fe1_task = PythonOperator(
    task_id='feature_engineering_1',
    python_callable=feature_engineering_1,
    provide_context=True,
    op_kwargs={'file_path': '/tmp/heart_disease_subset.csv'},
    dag=dag,
)

fe2_task = PythonOperator(
    task_id='feature_engineering_2',
    python_callable=feature_engineering_2,
    provide_context=True,
    op_kwargs={'file_path': '/tmp/heart_disease_subset.csv'},
    dag=dag,
)

train_model_1_task = PythonOperator(
    task_id='train_model_1',
    python_callable=train_models,
    provide_context=True,
    op_kwargs={'file_path': '/tmp/heart_disease_subset_fe1.csv'},
    dag=dag,
)

train_model_2_task = PythonOperator(
    task_id='train_model_2',
    python_callable=train_models,
    provide_context=True,
    op_kwargs={'file_path': '/tmp/heart_disease_subset_fe2.csv'},
    dag=dag,
)

# Define task dependencies
load_data_task >> eda_task >> [fe1_task, fe2_task]
fe1_task >> train_model_1_task
fe2_task >> train_model_2_task
