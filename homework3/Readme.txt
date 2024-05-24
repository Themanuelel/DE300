The data_cleaning_analysis.ipynb has all the cells that clean and impute the data including the webscraping, along with the binary classification model construction and analysis. 

The command I used to open the folder in my browser was:
docker run -p 8888:8888 -v $(pwd):/home/jovyan -v $HOME/.aws:/root/.aws jupiter-img

To build and open the container, run 'bash run.sh' 
You can start the container with 'docker start spark-sql-container'. Changing 'start' with 'stop' will stop the container.
'docker exec -it spark-sql-container /bin/bash' will open the container
To open the container's files in browser, use 'jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root' 
Since pyspark is not installed in my ec2 instance, you will only be able to run the cells within the spark-sql-container.
ctrl-c will exit the notebook in termimal, and 'exit' will exit the container
You cannot have the instance's files and the container's files open in browser at the same time, so stopping the container will allow you to open the instance files
in browser if needed.

After adding the two imputed smoke columns, the original smoke column was removed since it wasn't needed and didn't allow me to perform my model analysis.
