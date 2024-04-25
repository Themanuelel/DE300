import pandas as pd
from sqlalchemy import create_engine, text

# Function to determine appropriate data type for each column
def determine_data_type(series):
    if pd.api.types.is_integer_dtype(series):
        return 'INTEGER'
    elif pd.api.types.is_float_dtype(series):
        return 'REAL'
#    elif pd.api.types.is_string_dtype(series):
#        return 'VARCHAR({})'.format(series.str.len().max())
#    else:
#        return 'VARCHAR(255)'  # Default to VARCHAR if data type is not recognized

# Read CSV file
csv_file_path = 'heart_disease.csv'
df = pd.read_csv(csv_file_path)

# Analyze sample data to determine data types for each column
column_data_types = {col: determine_data_type(df[col]) for col in df.columns}

# Generate SQL code to create table
table_name = 'another_table'
create_table_sql = f'CREATE TABLE IF NOT EXISTS {table_name} (\n'
for col, data_type in column_data_types.items():
    create_table_sql += f'    {col} {data_type},\n'
create_table_sql = create_table_sql.rstrip(',\n') + '\n);'

# Connect to the database using SQLAlchemy
db_url = 'postgresql://postgres:password@localhost:5432/hearts'
engine = create_engine(db_url)

# Create table in the database
with engine.connect() as conn:
    conn.execute(text(create_table_sql))

# Load data into the database table
df.to_sql(table_name, engine, if_exists='append', index=False)

print('Table created and data loaded successfully.')
