
'docker run -p 8888:8888 -v $(pwd):/home/jovyan -v $HOME/.aws:/root/.aws jupiter-img' will allow for jupyter browser editing using localhost
Assuming python3 installed, run 'python3 boxplots.py' to generate the boxplots. 
Boxplots.py also contains comments that analyse the results and provides reasoning for any imputation. 
Running 'python3 connect.py' will load the heart_disease.csv file into my database
Running 'python3 analysis.py' will perform EDA on the statistics and provide visualizations. 
