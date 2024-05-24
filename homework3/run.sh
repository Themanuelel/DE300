docker run -v /home/ubuntu/DE300/homework3:/tmp/homework3 -it \
           -p 8888:8888 \
           --name spark-sql-container \
	   pyspark-image