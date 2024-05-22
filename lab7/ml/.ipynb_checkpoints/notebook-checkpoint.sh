#!/bin/bash

# Remove previous output directory if it exists
/bin/rm -r -f counts

# Set the Python interpreter for PySpark
export PYSPARK_PYTHON=../demos/bin/python3

# Convert the Jupyter notebook to a Python script
jupyter nbconvert --to script ml_pyspark.ipynb

# Run the Python script using Spark-submit
/opt/spark/bin/spark-submit --archives ../demos.tar.gz#demos ml_pyspark.py
