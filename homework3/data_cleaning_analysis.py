#!/usr/bin/env python
# coding: utf-8

# In[2]:


from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, mean, median

# Create a SparkSession
spark = SparkSession.builder \
    .appName("Heart Disease Analysis") \
    .getOrCreate()

# Read the CSV file into a Spark DataFrame
df = spark.read.csv("heart_disease.csv", header=True, inferSchema=True)

df = df.limit(899)

# Retain only the specified columns
columns_to_keep = ['age', 'sex', 'painloc', 'painexer', 'cp', 'trestbps', 
                   'smoke', 'fbs', 'prop', 'nitr', 'pro', 'diuretic', 
                   'thaldur', 'thalach', 'exang', 'oldpeak', 'slope', 'target']

df_subset = df.select(columns_to_keep)

## Replaces all values for painloc and painexer with the mode value for those columns
painloc_mode = df_subset.groupby().agg({"painloc": "max"}).collect()[0][0]
painexer_mode = df_subset.groupby().agg({"painexer": "max"}).collect()[0][0]

df_subset = df_subset.fillna({"painloc": painloc_mode, "painexer": painexer_mode})

## Values < 100 are replaced with 100
df_subset = df_subset.withColumn("trestbps", when(col("trestbps") < 100, 100).otherwise(col("trestbps")))

## Values less than 0 are replaced with 0, those greater than 4 are replaced with 4
df_subset = df_subset.withColumn("oldpeak", when(col("oldpeak") < 0, 0).otherwise(col("oldpeak")))
df_subset = df_subset.withColumn("oldpeak", when(col("oldpeak") > 4, 4).otherwise(col("oldpeak")))

## Filling missing values with the mean
mean_thaldur = df_subset.select(mean("thaldur")).collect()[0][0]
mean_thalach = df_subset.select(mean("thalach")).collect()[0][0]

df_subset = df_subset.fillna({"thaldur": mean_thaldur, "thalach": mean_thalach})

## Filling missing values with the mode value
mode_fbs = df_subset.groupby().agg({"fbs": "max"}).collect()[0][0]
mode_prop = df_subset.groupby().agg({"prop": "max"}).collect()[0][0]
mode_nitr = df_subset.groupby().agg({"nitr": "max"}).collect()[0][0]
mode_pro = df_subset.groupby().agg({"pro": "max"}).collect()[0][0]
mode_diuretic = df_subset.groupby().agg({"diuretic": "max"}).collect()[0][0]
mode_exang = df_subset.groupby().agg({"exang": "max"}).collect()[0][0]
mode_slope = df_subset.groupby().agg({"slope": "max"}).collect()[0][0]

df_subset = df_subset.fillna({"fbs": mode_fbs, "prop": mode_prop, "nitr": mode_nitr, 
                               "pro": mode_pro, "diuretic": mode_diuretic, 
                               "exang": mode_exang, "slope": mode_slope})

## Also replaces values greater than 1 with the mode for that column
df_subset = df_subset.withColumn("fbs", when(col("fbs") > 1, mode_fbs).otherwise(col("fbs")))
df_subset = df_subset.withColumn("prop", when(col("prop") > 1, mode_prop).otherwise(col("prop")))
df_subset = df_subset.withColumn("nitr", when(col("nitr") > 1, mode_nitr).otherwise(col("nitr")))
df_subset = df_subset.withColumn("pro", when(col("pro") > 1, mode_pro).otherwise(col("pro")))
df_subset = df_subset.withColumn("diuretic", when(col("diuretic") > 1, mode_diuretic).otherwise(col("diuretic")))

## These columns are checked for skewness. If they appear to be skewed, the missing values are filled with
## the median. If not skewed, the missing values are filled with the mean. 
subs_cols = ['trestbps', 'oldpeak', 'thaldur', 'thalach']

skewness = {col_name: df_subset.agg({col_name: "skewness"}).collect()[0][0] for col_name in subs_cols}

for col_name, col_skewness in skewness.items():
    if abs(col_skewness) < 0.5:
        # If not skewed, replace missing values with mean
        mean_val = df_subset.select(mean(col_name)).collect()[0][0]
        df_subset = df_subset.fillna({col_name: mean_val})
    else:
        # If skewed, replace missing values with median
        median_val = df_subset.approxQuantile(col_name, [0.5], 0.25)[0]
        df_subset = df_subset.fillna({col_name: median_val})

# Save the modified DataFrame to a new CSV file
df_subset.write.mode("overwrite").csv("heart_disease_subset.csv", header=True)


# In[10]:


# print(df.shape)


# In[3]:


get_ipython().system('pip install scrapy')


# In[5]:


import os
import requests
from scrapy import Selector
from pathlib import Path
import re
from typing import List
from pyspark.sql import SparkSession
from pyspark.sql.functions import col


# Constants
DATA_FOLDER = Path('data/')
URL = 'https://www.abs.gov.au/statistics/health/health-conditions-and-risks/smoking/latest-release'

if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

def get_selector_from_url(url: str) -> Selector:
    response = requests.get(url)
    return Selector(text=response.content)

def parse_row(row: Selector) -> List[str]:
    '''
    Parses an HTML row into a list of individual elements
    '''
    cells = row.xpath('.//th | .//td')
    row_data = []
    
    for cell in cells:
        cell_text = cell.xpath('normalize-space(.)').get()
        cell_text = re.sub(r'<.*?>', ' ', cell_text)  # Remove remaining HTML tags
        cell_text = cell_text.replace('\xa0', '')  # Remove \xa0 characters
        row_data.append(cell_text)
    
    return row_data

def parse_table_as_list(table_sel: Selector, header: bool = True) -> (List[str], List[List[str]]):
    '''
    Parses an HTML table and returns it as a list of lists
    '''
    # Extract rows
    rows = table_sel.xpath('./tbody//tr')
    
    # Parse header and the remaining rows
    columns = None
    if header:
        columns = parse_row(rows[0])
        rows = rows[1:]  # Skip the header row
    
    table_data = [parse_row(row) for row in rows]
    
    return columns, table_data

selector = get_selector_from_url(URL)

# Select the table containing smoking data by age
smoking_table = selector.xpath('//table[caption[contains(text(),"Proportion of people 15 years and over who were current daily smokers by age")]]')

if smoking_table:
    try:
        columns, table_data = parse_table_as_list(smoking_table[0], header=True)
    except Exception as e:
        print(f"Error: {e}")
    else:
        # Create a Spark DataFrame from the parsed data
        spark_df = spark.createDataFrame(table_data, schema=columns)
        
        # Select only the first and specific column (11th column in this case, considering zero-based indexing)
        # Adjust the column index if the desired column is different
        selected_df = spark_df.select(col(columns[0]), col(f"`{columns[10]}`"))
        
        # Show the selected DataFrame
        selected_df.show(truncate=False)
else:
    print("Smoking table not found on the webpage.")


# In[18]:


import os
import requests
from scrapy import Selector
from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_extract

# Constants
URL = 'https://www.cdc.gov/tobacco/data_statistics/fact_sheets/adult_data/cig_smoking/index.htm'

def get_selector_from_url(url: str) -> Selector:
    response = requests.get(url)
    return Selector(text=response.content)

# Fetch and parse the HTML content
selector = get_selector_from_url(URL)

# Extract the paragraphs containing the desired text
paragraphsM = selector.xpath("//li[contains(text(), 'adult men')]/text()").get()
paragraphsF = selector.xpath("//li[contains(text(), 'adult women')]/text()").get()

# Print extracted paragraphs
if paragraphsM:
    print(paragraphsM.strip())
if paragraphsF:
    print(paragraphsF.strip())

# Extract list items following a specific header
uls = selector.xpath("//div[h4[contains(text(), 'By Age')]]/following-sibling::div//ul/li/text()").getall()

# Print extracted list items
for line in uls:
    print(line.strip())

# Prepare data for Spark DataFrame
data = []
if paragraphsM:
    data.append(("adult men", paragraphsM.strip()))
if paragraphsF:
    data.append(("adult women", paragraphsF.strip()))
for line in uls:
    data.append(("By Age", line.strip()))

# Define schema
columns = ["Category", "Text"]

# Create a Spark DataFrame
df = spark.createDataFrame(data, schema=columns)

# Show the DataFrame
df.show(truncate=False)


# In[3]:


from pyspark.sql.functions import col, when, udf
from pyspark.sql.types import FloatType

# Define the function to get the smoking percentage based on age
def get_smoking_percentage(age: int) -> float:
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
        return None  # Return None for unknown age ranges

# Convert the function to a UDF
get_smoking_udf = udf(lambda age: get_smoking_percentage(age), FloatType())

# Convert the 'age' column to integer type
df_subset = df_subset.withColumn('age', col('age').cast('int'))

# Apply the function to create a new column 'smoking_src1' with the updated values
df_subset = df_subset.withColumn('smoking_src1', 
                                 when(col('smoke').isin([0, 1]), col('smoke').cast(FloatType()))
                                 .otherwise(get_smoking_udf(col('age'))))

# Save the modified DataFrame to the same CSV file, overwriting the existing file
df_subset.write.csv('heart_disease_subset.csv', header=True, mode='overwrite')


# In[4]:


from pyspark.sql.functions import col, when, udf
from pyspark.sql.types import FloatType

# Define the function to get the smoking percentage based on age and sex
def get_smoking_percentage(age: int, sex: int) -> float:
    if sex == 0:  # Female
        if 18 <= age <= 24:
            return .053
        elif 25 <= age <= 44:
            return .126
        elif 45 <= age <= 64:
            return .149
        elif age >= 65:
            return .083
    elif sex == 1:  # Male
        if 18 <= age <= 24:
            return round(.053 * (.131 / .101), 3)
        elif 25 <= age <= 44:
            return round(.126 * (.131 / .101), 3)
        elif 45 <= age <= 64:
            return round(.149 * (.131 / .101), 3)
        elif age >= 65:
            return round(.083 * (.131 / .101), 3)
    return None

# Convert the function to a UDF
get_smoking_udf = udf(lambda age, sex: get_smoking_percentage(age, sex), FloatType())

# Convert the 'age' and 'sex' columns to integer type
df_subset = df_subset.withColumn('age', col('age').cast('int'))
df_subset = df_subset.withColumn('sex', col('sex').cast('int'))

# Apply the function to create a new column 'smoke_src2' with the updated values
df_subset = df_subset.withColumn('smoke_src2', 
                                 when(col('smoke').isin([0, 1]), col('smoke').cast(FloatType()))
                                 .otherwise(get_smoking_udf(col('age'), col('sex'))))

# Save the modified DataFrame to the same CSV file, overwriting the existing file
df_subset.write.csv('heart_disease_subset.csv', header=True, mode='overwrite')


# In[5]:


# Remove the 'smoke' column
df_subset = df_subset.drop('smoke')

# Save the modified DataFrame to the same CSV file, overwriting the existing file
df_subset.write.csv('heart_disease_subset.csv', header=True, mode='overwrite')


# In[8]:


from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline

# Load the dataset
df = spark.read.csv('heart_disease_subset.csv', header=True, inferSchema=True)

# Define the feature columns
feature_cols = df.columns
feature_cols.remove('target')

# Create separate instances of VectorAssembler for each pipeline
logistic_assembler = VectorAssembler(inputCols=feature_cols, outputCol='features_logistic')
random_forest_assembler = VectorAssembler(inputCols=feature_cols, outputCol='features_random_forest')
svm_assembler = VectorAssembler(inputCols=feature_cols, outputCol='features_svm')

# Split the data into training and test sets
train_df, test_df = df.randomSplit([0.9, 0.1], seed=42)

# Initialize the models
logistic_model = LogisticRegression(labelCol='target', featuresCol='features_logistic', maxIter=1000)
random_forest_model = RandomForestClassifier(labelCol='target', featuresCol='features_random_forest', maxDepth=10)
svm_model = LinearSVC(labelCol='target', featuresCol='features_svm', maxIter=1000)

# Create pipelines for each model
logistic_pipeline = Pipeline(stages=[logistic_assembler, logistic_model])
random_forest_pipeline = Pipeline(stages=[random_forest_assembler, random_forest_model])
svm_pipeline = Pipeline(stages=[svm_assembler, svm_model])

# Train the models
logistic_model = logistic_pipeline.fit(train_df)
random_forest_model = random_forest_pipeline.fit(train_df)
svm_model = svm_pipeline.fit(train_df)

# Make predictions on the test data
logistic_pred = logistic_model.transform(test_df)
random_forest_pred = random_forest_model.transform(test_df)
svm_pred = svm_model.transform(test_df)

# Evaluate the performance of the models
evaluator = MulticlassClassificationEvaluator(labelCol='target', metricName='accuracy')

logistic_accuracy = evaluator.evaluate(logistic_pred)
random_forest_accuracy = evaluator.evaluate(random_forest_pred)
svm_accuracy = evaluator.evaluate(svm_pred)

# Print the accuracy of each model
print(f"Logistic Regression Model Accuracy: {logistic_accuracy:.4f}")
print(f"Random Forest Model Accuracy: {random_forest_accuracy:.4f}")
print(f"SVM Model Accuracy: {svm_accuracy:.4f}")

# Perform 5-fold cross-validation
paramGrid = ParamGridBuilder().build()

crossval = CrossValidator(estimator=logistic_pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=5)

cv_logistic_model = crossval.fit(train_df)
logistic_cv_score = cv_logistic_model.avgMetrics[0]

crossval = CrossValidator(estimator=random_forest_pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=5)

cv_random_forest_model = crossval.fit(train_df)
random_forest_cv_score = cv_random_forest_model.avgMetrics[0]

crossval = CrossValidator(estimator=svm_pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=5)

cv_svm_model = crossval.fit(train_df)
svm_cv_score = cv_svm_model.avgMetrics[0]

# Print mean cross-validation scores
print(f"Logistic Regression Model Mean Cross-Validation Score: {logistic_cv_score:.4f}")
print(f"Random Forest Model Mean Cross-Validation Score: {random_forest_cv_score:.4f}")
print(f"SVM Model Mean Cross-Validation Score: {svm_cv_score:.4f}")


# In[ ]:


### ANALYSIS
# Based on the results, I would choose the random forest model again. 
# This is an easier decision than last time, since now the random forest model has both a higher accuracy and a higher mean cross-validation score than
# the other two models. This means that the random forest model performed better on the test data and is more likely to do well when testing on new data. 

