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

df_filtered = df[df['trestbps'] != 0]
df_filtered = df_filtered[df_filtered['chol'] != 0]

#### AGE VS BLOOD PRESSURE
plt.figure(figsize=(10, 6))

plt.scatter(df_filtered['age'], df_filtered['trestbps'], alpha=0.5)
plt.title('Scatterplot of Age vs. trestbps')
plt.xlabel('Age')
plt.ylabel('trestbps')

# Save the plot as an image file
plt.savefig('age_vs_trestbps_scatterplot.png')

plt.close()


#### AGE VS CHOLESTEROL

plt.figure(figsize=(10, 6))

plt.scatter(df_filtered['age'], df_filtered['chol'], alpha=0.5)
plt.title('Scatterplot of Age vs. Cholesterol')
plt.xlabel('Age')
plt.ylabel('Cholesterol')

# Save the plot as an image file
plt.savefig('age_vs_chol_scatterplot.png')

plt.close()


#### GENDER VS BLOOD PRESSURE

# Separating male and female
male_trestbps = df_filtered[df_filtered['sex'] == 1]['trestbps']
female_trestbps = df_filtered[df_filtered['sex'] == 0]['trestbps']

mean_male_trestbps = male_trestbps.mean()
mean_female_trestbps = female_trestbps.mean()

# Create histograms for trestbps comparing males and females
plt.figure(figsize=(10, 6))

plt.hist(male_trestbps, alpha=0.5, color='blue', label='Male')
plt.hist(female_trestbps, alpha=0.5, color='pink', label='Female')

# Add mean lines to the plot
plt.axvline(mean_male_trestbps, color='blue', linestyle='dashed', linewidth=1, label=f'Mean trestbps (Male): {mean_male_trestbps:.2f}')
plt.axvline(mean_female_trestbps, color='pink', linestyle='dashed', linewidth=1, label=f'Mean trestbps (Female): {mean_female_trestbps:.2f}')

plt.title('Histogram of Resting Blood Pressure (trestbps) by Sex')
plt.xlabel('Resting Blood Pressure (trestbps)')
plt.ylabel('Frequency')

plt.legend()

# Save the plot as an image file
plt.savefig('trestbps_histogram_by_sex.png')

plt.close()


#### SMOKING? VS BLOOD PRESSURE

# Separate the DataFrame into two groups: smokers and nonsmokers
smokers = df_filtered[df_filtered['smoke'] == 1]
nonsmokers = df_filtered[df_filtered['smoke'] == 0]

# Plot histograms for trestbps among smokers and nonsmokers
plt.figure(figsize=(12, 6))

plt.hist(smokers['trestbps'], bins=20, alpha=0.5, label='Smokers')
plt.hist(nonsmokers['trestbps'], bins=20, alpha=0.5, label='Nonsmokers')

# Calculate mean trestbps for smokers and nonsmokers
mean_trestbps_smokers = smokers['trestbps'].mean()
mean_trestbps_nonsmokers = nonsmokers['trestbps'].mean()

# Add mean lines to the plot
plt.axvline(mean_trestbps_smokers, color='blue', linestyle='dashed', linewidth=1, label=f'Mean trestbps (Smokers): {mean_trestbps_smokers:.2f}')
plt.axvline(mean_trestbps_nonsmokers, color='orange', linestyle='dashed', linewidth=1, label=f'Mean trestbps (Nonsmokers): {mean_trestbps_nonsmokers:.2f}')

plt.title('Distribution of trestbps among Smokers and Nonsmokers')
plt.xlabel('trestbps')
plt.ylabel('Frequency')
plt.legend()

# Save the plot as an image file
plt.savefig('trestbps_distribution_with_mean.png')

plt.close()