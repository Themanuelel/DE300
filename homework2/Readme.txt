the data_cleaning_analysis.ipynb has all the cells that clean and impute the data including the webscraping, along with the binary classification model construction and analysis. 

The command I used to open the folder in my browser was:
docker run -p 8888:8888 -v $(pwd):/home/jovyan -v $HOME/.aws:/root/.aws jupiter-img

Then connecting to the browser allowed me to run the files. 

In browser the heart_disease_subset.csv file sometimes doesn't update when running a cell that updates it. Even though the data itself is updated, it may still appear unchanged. In that case, deleting the heart_disease_subset.csv file and running the cell again will refresh the csv file. 

After adding the two imputed smoke columns, the original smoke column was removed since it wasn't needed and didn't allow me to perform my model analysis.

Since the instance sometimes crashes when running the last cell, I have included a sample output within the cell from running it on my local pc.