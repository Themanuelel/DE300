import mysql.connector
import pandas as pd
import matplotlib.pyplot as plt

# Establish a connection to the MySQL database
connection = mysql.connector.connect(
    host="localhost",
    user="new_user",
    password="",
    database="heart_database"
)

# Create a cursor object to execute SQL queries
cursor = connection.cursor()

# Execute a SQL query to fetch data
cursor.execute("SELECT * FROM third_table")

# Fetch all rows from the result set
rows = cursor.fetchall()

# Get column names
column_names = [desc[0] for desc in cursor.description]

# Create a DataFrame from the fetched data
df = pd.DataFrame(rows, columns=column_names)

# Close cursor and connection
cursor.close()
connection.close()

# Plot box plots for all columns
plt.figure(figsize=(16, 10))

# Get the number of columns to plot
num_cols = min(len(df.columns), 15)

for i in range(num_cols):
    plt.subplot(3, 5, i + 1)
    df.boxplot(column=df.columns[i])
    plt.title(df.columns[i])

plt.tight_layout()

# Save the plot to a single file
plt.savefig("boxplots.png")

plt.close()  # Close the figure to release resources

## For the case of missing values, they were filled in with 0s. I chose this approach because imputation with a mean or median value would potentially affect the results. In other cases, a row with significant empty values was removed, due to the amount of artifical data that would have to be added. 
## The boxplots show outliers for resting blood pressure and cholesterol. For further analysis, those rows with 0 values are filtered out. This is because it is impossible to have a blood pressure or cholesterol value of 0, unless the person is deceased. 
## Other plots that show significant outliers are catagorical, which means that they represent a skewed distribution of users. The 'sex' boxplot, for example, has a mean of 1 and every value of 0 is an outlier. This doesn't mean that the 0's shouldn't be included, it just means that there are more men reporting their results than women. 











