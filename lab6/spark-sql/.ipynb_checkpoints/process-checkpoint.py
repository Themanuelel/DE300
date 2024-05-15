from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType
from pyspark.sql.functions import col, when, isnan, isnull, count, avg, trim
import os

DATA_FOLDER = "data"
# source https://www.statista.com/statistics/242030/marital-status-of-the-us-population-by-sex/
# the first value is male and the second is for female
MARITAL_STATUS_BY_GENDER = [
    ["Never-married", 47.35, 41.81],
    ["Married-AF-spouse", 67.54, 68.33],
    ["Widowed", 3.58, 11.61],
    ["Divorced", 10.82, 15.09]
]
MARITAL_STATUS_BY_GENDER_COLUMNS = ["marital_status_statistics", "male", "female"]

def read_data(spark: SparkSession):
    """
    read data based on the given schema; this is much faster than spark determining the schema
    """
    schema = StructType([
        StructField("age", IntegerType(), True),
        StructField("workclass", StringType(), True),
        StructField("fnlwgt", FloatType(), True),
        StructField("education", StringType(), True),
        StructField("education_num", FloatType(), True),
        StructField("marital_status", StringType(), True),
        StructField("occupation", StringType(), True),
        StructField("relationship", StringType(), True),
        StructField("race", StringType(), True),
        StructField("sex", StringType(), True),
        StructField("capital_gain", FloatType(), True),
        StructField("capital_loss", FloatType(), True),
        StructField("hours_per_week", FloatType(), True),
        StructField("native_country", StringType(), True),
        StructField("income", StringType(), True)
    ])

    data = spark.read \
        .schema(schema) \
        .option("header", "false") \
        .option("inferSchema", "false") \
        .csv(os.path.join(DATA_FOLDER, "*.csv"))

    data = data.repartition(8)

    float_columns = [f.name for f in data.schema.fields if isinstance(f.dataType, FloatType)]
    for v in float_columns:
        data = data.withColumn(v, data[v].cast(IntegerType()))

    string_columns = [f.name for f in data.schema.fields if isinstance(f.dataType, StringType)]
    for column in string_columns:
        data = data.withColumn(column, trim(data[column]))

    data.show(5)

    return data

def missing_values(data):
    """
    count the number of samples with missing values for each row
    remove such samples
    """
    missing_values = data.select([count(when(isnan(c) | isnull(c), c)).alias(c) for c in data.columns])
    missing_values.show()

    num_samples = data.count()
    print("Number of samples:", num_samples)

    data = data.dropna()
    
    return data

def feature_engineering(data):
    """
    calculate the product of each pair of integer features
    """
    integer_columns = [f.name for f in data.schema.fields if isinstance(f.dataType, IntegerType)]
    for i, col1 in enumerate(integer_columns):
        for col2 in integer_columns[i:]:
            product_col_name = f"{col1}_x_{col2}"
            data = data.withColumn(product_col_name, col(col1) * col(col2))

    data.show(5)

    return data

def bias_marital_status(data):
    """
    is there bias in capital gain by marital status
    """
    average_capital_gain = data.groupBy("marital_status").agg(avg("capital_gain").alias("average_capital_gain"))
    average_capital_gain.show()

    divorced_data = data.filter(data.marital_status == "Divorced")
    divorced_data.show(5)

def join_with_US_gender(spark, data):
    """
    join with respect to the marital_status
    """
    us_df = spark.createDataFrame(MARITAL_STATUS_BY_GENDER, MARITAL_STATUS_BY_GENDER_COLUMNS)
    return data.join(us_df, data.marital_status == us_df.marital_status_statistics, 'outer')

def main():
    spark = SparkSession.builder \
        .appName("Read Adult Dataset") \
        .getOrCreate()

    data = read_data(spark)
    data = missing_values(data)
    data = feature_engineering(data)
    bias_marital_status(data)
    data = join_with_US_gender(spark, data)

    data.show(5)
    data.write.format('csv').option('header', 'true').mode('overwrite').save(os.path.join(DATA_FOLDER, 'processed_data.csv'))

    spark.stop()

if __name__ == "__main__":
    main()
