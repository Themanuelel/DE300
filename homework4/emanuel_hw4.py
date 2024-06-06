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
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, mean, median, monotonically_increasing_id
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.sql.types import FloatType
from pyspark.sql.functions import udf

# Define default_args for the DAG
default_args = {
    'owner': 'Emanuel',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

# Define the DAG
dag = DAG(
    'eman_hw4',
    default_args=default_args,
    description='eman_hw4',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2023, 1, 1),
    catchup=False,
)

def load_data_from_s3(**kwargs):
    s3 = boto3.client('s3')
    bucket_name = 'de300spring2024-eman'
    key = 'data/heart_disease.csv'
    local_dir = '/tmp'  # Use a writable directory path
    local_path = f"{local_dir}/heart_disease.csv"
    s3.download_file(bucket_name, key, local_path)
    # s3.download_file(bucket_name, key, '/tmp/spark_heart_disease.csv')
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

def create_spark_session():
    spark = SparkSession.builder \
        .appName("Heart Disease Analysis") \
        .getOrCreate()
    return spark


def spark_eda_preprocessing(input_path, output_path, **kwargs):
    spark = create_spark_session()
    df = spark.read.csv(input_path, header=True, inferSchema=True)
    df = df.limit(899)

    columns_to_keep = ['age', 'sex', 'painloc', 'painexer', 'cp', 'trestbps', 
                       'smoke', 'fbs', 'prop', 'nitr', 'pro', 'diuretic', 
                       'thaldur', 'thalach', 'exang', 'oldpeak', 'slope', 'target']
    df_subset = df.select(columns_to_keep)

    # Fill missing values
    from pyspark.sql.functions import col, when, mean, round as spark_round, expr, mode as spark_mode
    from pyspark.sql import functions as F

    # Handle categorical missing values
    mode_impute_cols = ['painloc', 'painexer', 'fbs', 'prop', 'nitr', 'pro', 'diuretic', 'exang', 'slope']
    for col_name in mode_impute_cols:
        mode_value = df_subset.groupBy(col_name).count().orderBy(F.desc('count')).first()[0]
        df_subset = df_subset.fillna({col_name: mode_value})

    # Cap and floor values
    df_subset = df_subset.withColumn('trestbps', when(col('trestbps') < 100, 100).otherwise(col('trestbps')))
    df_subset = df_subset.withColumn('oldpeak', when(col('oldpeak') < 0, 0).when(col('oldpeak') > 4, 4).otherwise(col('oldpeak')))

    # Impute mean or median for continuous variables based on skewness
    mean_thaldur = df_subset.agg(spark_round(mean('thaldur'), 1)).first()[0]
    mean_thalach = df_subset.agg(spark_round(mean('thalach'), 1)).first()[0]
    df_subset = df_subset.fillna({'thaldur': mean_thaldur, 'thalach': mean_thalach})

    # Replace outliers with mode values
    outlier_cols = {'fbs': mode_value, 'prop': mode_value, 'nitr': mode_value, 'pro': mode_value, 'diuretic': mode_value}
    for col_name, mode_value in outlier_cols.items():
        df_subset = df_subset.withColumn(col_name, when(col(col_name) > 1, mode_value).otherwise(col(col_name)))

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

    smoking_src1_udf = F.udf(get_smoking_percentage_src1)
    df_subset = df_subset.withColumn('smoking_src1', when(col('smoke').isin([0, 1]), col('smoke')).otherwise(smoking_src1_udf(col('age'))))

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

    smoking_src2_udf = F.udf(get_smoking_percentage_src2)
    df_subset = df_subset.withColumn('smoke_src2', when(col('smoke').isin([0, 1]), col('smoke')).otherwise(smoking_src2_udf(col('age'), col('sex'))))

    # Drop the original 'smoke' column
    df_subset = df_subset.drop('smoke')

    # Save the modified DataFrame
    df_subset.write.csv(output_path, header=True, mode='overwrite')
    return output_path


def spark_feature_engineering(input_path, output_path, **kwargs):
    spark = create_spark_session()
    df = spark.read.csv(input_path, header=True, inferSchema=True)

    # Example feature engineering: square root of trestbps
    df = df.withColumn('trestbps_sqrt', col('trestbps') ** 0.5)
    df = df.withColumn('id', monotonically_increasing_id())

    # Save the modified DataFrame
    fe_path = output_path.replace('.csv', '_d.csv')
    df.write.csv(fe_path, header=True, mode='overwrite')
    ti = kwargs['ti']
    ti.xcom_push(key='fe_spark_path', value=fe_path)
    return fe_path


def spark_train_model_svm(**kwargs):
    # Load the dataset
    spark = create_spark_session()
    df = spark.read.csv('/tmp/heart_disease_subset_fe_spark_d.csv', header=True, inferSchema=True)

    # Define the feature columns
    feature_cols = df.columns
    feature_cols.remove('target')

    # Create separate instances of VectorAssembler for each pipeline
    assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')

    # Split the data into training and test sets
    train_df, test_df = df.randomSplit([0.9, 0.1], seed=42)

    # Initialize the model
    model = pyspark.ml.classification.LinearSVC(labelCol='target', featuresCol='features', maxIter=100)

    # Create a pipeline for the model
    pipeline = Pipeline(stages=[assembler, model])

    # Train the model
    model = pipeline.fit(train_df)

    # Make predictions on the test data
    predictions = model.transform(test_df)

    # Evaluate the performance of the model
    evaluator = MulticlassClassificationEvaluator(labelCol='target', metricName='accuracy')
    accuracy = evaluator.evaluate(predictions)

    # Print the accuracy of the model
    print(f"SVM Model Accuracy: {accuracy:.4f}")


    ti = kwargs['ti']
    ti.xcom_push(key='svm_accuracy', value=accuracy)


def spark_train_model_logistic_regression(**kwargs):
    # Load the dataset
    spark = create_spark_session()
    df = spark.read.csv('/tmp/heart_disease_subset_fe_spark_d.csv', header=True, inferSchema=True)

    # Define the feature columns
    feature_cols = df.columns
    feature_cols.remove('target')

    # Create separate instances of VectorAssembler for each pipeline
    assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')

    # Split the data into training and test sets
    train_df, test_df = df.randomSplit([0.9, 0.1], seed=42)

    # Initialize the model
    model = pyspark.ml.classification.LogisticRegression(labelCol='target', featuresCol='features', maxIter=1000)

    # Create a pipeline for the model
    pipeline = Pipeline(stages=[assembler, model])

    # Train the model
    model = pipeline.fit(train_df)

    # Make predictions on the test data
    predictions = model.transform(test_df)

    # Evaluate the performance of the model
    evaluator = MulticlassClassificationEvaluator(labelCol='target', metricName='accuracy')
    accuracy = evaluator.evaluate(predictions)

    # Print the accuracy of the model
    print(f"Logistic Regression Model Accuracy: {accuracy:.4f}")


    ti = kwargs['ti']
    ti.xcom_push(key='lr_accuracy', value=accuracy)


# Function for feature engineering strategy 1
def feature_engineering(file_path, **kwargs):
    df = pd.read_csv(file_path)
    
    # Example feature engineering: square of age
    df['age_squared'] = df['age'] ** 2
    df['id'] = df.index  # Add a unique identifier
    # Save the modified DataFrame
    fe_path = file_path.replace('.csv', '_fe.csv')
    df.to_csv(fe_path, index=False)
    ti = kwargs['ti']
    ti.xcom_push(key='fe_path', value=fe_path)
    return fe_path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_model_svm(file_path, **kwargs):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Define the feature columns and target
    feature_cols = df.columns.tolist()
    feature_cols.remove('target')
    X = df[feature_cols]
    y = df['target']
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    # Train and evaluate SVM
    svm_model = SVC(random_state=42)
    svm_model.fit(X_train, y_train)
    svm_predictions = svm_model.predict(X_test)
    svm_accuracy = accuracy_score(y_test, svm_predictions)
    print(f"SVM Model Accuracy: {svm_accuracy:.4f}")


    ti = kwargs['ti']
    ti.xcom_push(key='svm_accuracy', value=svm_accuracy)



def train_model_logistic_regression(file_path, **kwargs):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Define the feature columns and target
    feature_cols = df.columns.tolist()
    feature_cols.remove('target')
    X = df[feature_cols]
    y = df['target']
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    # Train and evaluate LogisticRegression
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    lr_predictions = lr_model.predict(X_test)
    lr_accuracy = accuracy_score(y_test, lr_predictions)
    print(f"Logistic Regression Model Accuracy: {lr_accuracy:.4f}")

    ti = kwargs['ti']
    ti.xcom_push(key='lr_accuracy', value=lr_accuracy)


def scrape_merge(**kwargs):
    ti = kwargs['ti']
    fe_path = ti.xcom_pull(task_ids='feature_engineering', key='fe_path')
    fe_spark_path = ti.xcom_pull(task_ids='spark_feature_engineering', key='fe_spark_path')

    # Load both datasets
    df_sklearn = pd.read_csv(fe_path)
    spark = create_spark_session()
    df_spark = spark.read.csv(fe_spark_path, header=True, inferSchema=True).toPandas()

    # Ensure consistent column names
    df_spark.columns = [col.replace('"', '') for col in df_spark.columns]

    # Identify duplicate columns
    duplicate_cols = set(df_sklearn.columns).intersection(set(df_spark.columns)) - {'id'}

    # Remove duplicate columns from one of the DataFrames before merging
    df_spark = df_spark.drop(columns=duplicate_cols)

    # Merge datasets on a common column, assuming 'id' as a common column
    df_merged = pd.merge(df_sklearn, df_spark, on='id', how='inner')

    # Save merged dataset
    merged_path = '/tmp/heart_disease_merged_fe.csv'
    df_merged.to_csv(merged_path, index=False)
    ti.xcom_push(key='merged_path', value=merged_path)
    return merged_path


def train_merge_svm(**kwargs):
    ti = kwargs['ti']
    merged_path = ti.xcom_pull(task_ids='scrape_merge', key='merged_path')
    df = pd.read_csv(merged_path)
    feature_cols = df.columns.tolist()
    feature_cols.remove('target')
    X = df[feature_cols]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    svm_model = SVC()
    svm_model.fit(X_train, y_train)
    svm_predictions = svm_model.predict(X_test)
    svm_accuracy = accuracy_score(y_test, svm_predictions)
    print(f"SVM Model Accuracy: {svm_accuracy:.4f}")
    ti.xcom_push(key='svm_accuracy', value=svm_accuracy)


def train_merge_logistic_regression(**kwargs):
    ti = kwargs['ti']
    merged_path = ti.xcom_pull(task_ids='scrape_merge', key='merged_path')
    df = pd.read_csv(merged_path)
    feature_cols = df.columns.tolist()
    feature_cols.remove('target')
    X = df[feature_cols]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_predictions = lr_model.predict(X_test)
    lr_accuracy = accuracy_score(y_test, lr_predictions)
    print(f"Logistic Regression Model Accuracy: {lr_accuracy:.4f}")
    ti.xcom_push(key='lr_accuracy', value=lr_accuracy)


def compare_models(**kwargs):
    ti = kwargs['ti']
    svm_accuracy_sklearn = ti.xcom_pull(task_ids='train_model_svm', key='svm_accuracy')
    lr_accuracy_sklearn = ti.xcom_pull(task_ids='train_model_logistic_regression', key='lr_accuracy')
    svm_accuracy_spark = ti.xcom_pull(task_ids='spark_train_model_svm', key='svm_accuracy')
    lr_accuracy_spark = ti.xcom_pull(task_ids='spark_train_model_logistic_regression', key='lr_accuracy')
    svm_accuracy_merge = ti.xcom_pull(task_ids='train_merge_svm', key='svm_accuracy')
    lr_accuracy_merge = ti.xcom_pull(task_ids='train_merge_logistic_regression', key='lr_accuracy')

    accuracies = {
        'train_model_svm': svm_accuracy_sklearn,
        'train_model_logistic_regression': lr_accuracy_sklearn,
        'spark_train_model_svm': svm_accuracy_spark,
        'spark_train_model_logistic_regression': lr_accuracy_spark,
        'train_merge_svm': svm_accuracy_merge,
        'train_merge_logistic_regression': lr_accuracy_merge
    }

    best_model_task_id = max(accuracies, key=accuracies.get)
    ti.xcom_push(key='best_model_task_id', value=best_model_task_id)
    ti.xcom_push(key='best_model_accuracy', value=accuracies[best_model_task_id])
    

def evaluate_best_model(**kwargs):
    spark = create_spark_session()
    df = spark.read.csv('/tmp/heart_disease_subset.csv', header=True, inferSchema=True)
    feature_cols = df.columns
    feature_cols.remove('target')
    assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
    train_df, test_df = df.randomSplit([0.9, 0.1], seed=42)

    ti = kwargs['ti']
    best_model_task_id = ti.xcom_pull(task_ids='compare_models', key='best_model_task_id')

    if best_model_task_id == 'train_model_svm':
        model = SVC()
    elif best_model_task_id == 'train_model_logistic_regression':
        model = LogisticRegression(max_iter=1000)
    elif best_model_task_id == 'spark_train_model_svm':
        model = LinearSVC(labelCol='target', featuresCol='features', maxIter=100)
    elif best_model_task_id == 'spark_train_model_logistic_regression':
        model = pyspark.ml.classification.LogisticRegression(labelCol='target', featuresCol='features', maxIter=1000)
    elif best_model_task_id == 'train_merge_svm':
        df = spark.read.csv('/tmp/heart_disease_merged_fe.csv', header=True, inferSchema=True)
        feature_cols = df.columns
        feature_cols.remove('target')
        assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
        train_df, test_df = df.randomSplit([0.9, 0.1], seed=42)
        model = SVC()
    else:
        df = spark.read.csv('/tmp/heart_disease_merged_fe.csv', header=True, inferSchema=True)
        feature_cols = df.columns
        feature_cols.remove('target')
        assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
        train_df, test_df = df.randomSplit([0.9, 0.1], seed=42)
        model = LogisticRegression(max_iter=1000)


    if 'spark' in best_model_task_id:
        pipeline = Pipeline(stages=[assembler, model])
        model = pipeline.fit(train_df)
        predictions = model.transform(test_df)
        evaluator = MulticlassClassificationEvaluator(labelCol='target', metricName='accuracy')
        accuracy = evaluator.evaluate(predictions)
    else:
        df = df.toPandas()
        X = df[feature_cols]
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

    return (f"Best Model ({best_model_task_id}) Accuracy: {accuracy:.4f}")


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

spark_eda_task = PythonOperator(
    task_id='spark_eda_preprocessing',
    python_callable=spark_eda_preprocessing,
    op_kwargs={'input_path': '/tmp/heart_disease.csv', 'output_path': '/tmp/heart_disease_subset_eda.csv'},
    dag=dag,
)


fe_task = PythonOperator(
    task_id='feature_engineering',
    python_callable=feature_engineering,
    provide_context=True,
    op_kwargs={'file_path': '/tmp/heart_disease_subset.csv'},
    dag=dag,
)

spark_fe_task = PythonOperator(
    task_id='spark_feature_engineering',
    python_callable=spark_feature_engineering,
    op_args=['/tmp/heart_disease_subset_eda.csv', '/tmp/heart_disease_subset_fe_spark.csv'],
    dag=dag,
)

merge_fe_task = PythonOperator(
    task_id='scrape_merge',
    python_callable=scrape_merge,
    provide_context=True,
    dag=dag,
)

train_merge_svm_task = PythonOperator(
    task_id='train_merge_svm',
    python_callable=train_merge_svm,
    provide_context=True,
    dag=dag,
)

train_merge_logistic_regression_task = PythonOperator(
    task_id='train_merge_logistic_regression',
    python_callable=train_merge_logistic_regression,
    provide_context=True,
    dag=dag,
)

train_model_svm_task = PythonOperator(
    task_id='train_model_svm',
    python_callable=train_model_svm,
    provide_context=True,
    op_kwargs={'file_path': '/tmp/heart_disease_subset_fe.csv'},
    dag=dag,
)

train_model_logistic_regression_task = PythonOperator(
    task_id='train_model_logistic_regression',
    python_callable=train_model_logistic_regression,
    provide_context=True,
    op_kwargs={'file_path': '/tmp/heart_disease_subset_fe.csv'},
    dag=dag,
)

spark_train_model_svm_task = PythonOperator(
    task_id='spark_train_model_svm',
    python_callable=spark_train_model_svm,
    dag=dag,
)


spark_train_model_logistic_regression_task = PythonOperator(
    task_id='spark_train_model_logistic_regression',
    python_callable=spark_train_model_logistic_regression,
    dag=dag,
)

compare_models_task = PythonOperator(
    task_id='compare_models',
    python_callable=compare_models,
    provide_context=True,
    dag=dag,
)

evaluate_best_model_task = PythonOperator(
    task_id='evaluate_best_model',
    python_callable=evaluate_best_model,
    provide_context=True,
    dag=dag,
)


# Define task dependencies
load_data_task >> [eda_task, spark_eda_task]

eda_task >> fe_task
fe_task >> [train_model_svm_task, train_model_logistic_regression_task, merge_fe_task]

spark_eda_task >> spark_fe_task
spark_fe_task >> [spark_train_model_svm_task, spark_train_model_logistic_regression_task, merge_fe_task]

merge_fe_task >> [train_merge_svm_task, train_merge_logistic_regression_task]

[train_model_svm_task, train_model_logistic_regression_task, spark_train_model_svm_task, spark_train_model_logistic_regression_task, train_merge_svm_task, train_merge_logistic_regression_task] >> compare_models_task
compare_models_task >> evaluate_best_model_task