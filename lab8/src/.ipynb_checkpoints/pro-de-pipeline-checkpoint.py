from airflow import DAG
from airflow.operators.python_operator import PythonOperator, BranchPythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.hooks.base_hook import BaseHook
from airflow.utils.dates import days_ago
from airflow.providers.amazon.aws.transfers.s3_to_local import S3ToLocalOperator
import tomli
import pathlib

# read the parameters from toml
CONFIG_FILE = "/root/configs/config.toml"

TABLE_NAMES = {
    "original_data": "wine",
    "clean_data": "wine_clean_data",
    "train_data": "wine_train_data",
    "test_data": "wine_test_data",
    "normalization_data": "normalization_values",
    "max_fe": "max_fe_features",
    "product_fe": "product_fe_features"
}

ENCODED_SUFFIX = "_encoded"
NORMALIZATION_TABLE_COLUMN_NAMES = ["name", "data_min", "data_max", "scale", "min"]

# Define the default args dictionary for DAG
default_args = {
    'owner': 'johndoe',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'retries': 0,
}

def read_config() -> dict:
    path = pathlib.Path(CONFIG_FILE)
    with path.open(mode="rb") as param_file:
        params = tomli.load(param_file)
    return params

PARAMS = read_config()

def create_db_connection():
    """
    create a db connection to the postgres connection

    return the connection
    """
    
    import re
    from sqlalchemy import create_engine

    conn = BaseHook.get_connection(PARAMS['db']['db_connection'])
    conn_uri = conn.get_uri()

    # replace the driver; airflow connections use postgres which needs to be replaced
    conn_uri= re.sub('^[^:]*://', PARAMS['db']['db_alchemy_driver']+'://', conn_uri)

    engine = create_engine(conn_uri)
    conn = engine.connect()

    return conn

def from_table_to_df(input_table_names: list[str], output_table_names: list[str]):
    """
    Decorator to open a list of tables input_table_names, load them in df and pass the dataframe to the function; on exit, it deletes tables in output_table_names
    The function has key = dfs with the value corresponding the list of the dataframes 

    The function must return a dictionary with key dfs; the values must be a list of dictionaries with keys df and table_name; Each df is written to table table_name
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            import pandas as pd

            """
            load tables to dataframes
            """
            if input_table_names is None:
                raise ValueError('input_table_names cannot be None')
            
            _input_table_names = None
            if isinstance(input_table_names, str):
                _input_table_names = [input_table_names]
            else:
                _input_table_names = input_table_names

            import pandas as pd
            
            print(f'Loading input tables to dataframes: {_input_table_names}')

            # open the connection
            conn = create_db_connection()

            # read tables and convert to dataframes
            dfs = []
            for table_name in _input_table_names:
                df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
                dfs.append(df)

            if isinstance(input_table_names, str):
                dfs = dfs[0]

            """
            call the main function
            """

            kwargs['dfs'] = dfs
            kwargs['output_table_names'] = output_table_names
            result = func(*args, **kwargs)

            """
            delete tables
            """

            print(f'Deleting tables: {output_table_names}')
            if output_table_names is None:
                _output_table_names = []
            elif isinstance(output_table_names, str):
                _output_table_names = [output_table_names]
            else:
                _output_table_names = output_table_names
            
            print(f"Dropping tables {_output_table_names}")
            for table_name in _output_table_names:
                conn.execute(f"DROP TABLE IF EXISTS {table_name}")

            """
            write dataframes in result to tables
            """

            for pairs in result['dfs']:
                df = pairs['df']
                table_name = pairs['table_name']
                df.to_sql(table_name, conn, if_exists="replace", index=False)
                print(f"Wrote to table {table_name}")

            conn.close()
            result.pop('dfs')

            return result
        return wrapper
    return decorator

def add_data_to_table_func(**kwargs):
    """
    insert data from local csv to a db table
    """

    import pandas as pd

    conn = create_db_connection()

    df = pd.read_csv(PARAMS['files']['local_file'], header=0)
    df.to_sql(TABLE_NAMES['original_data'], conn, if_exists="replace", index=False)

    conn.close()

    return {'status': 1}

@from_table_to_df(TABLE_NAMES['original_data'], None)
def clean_data_func(**kwargs):
    """
    data cleaning: drop none, remove outliers based on z-scores
    apply label encoding on categorical variables: assumption is that every string column is categorical
    """

    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    data_df = kwargs['dfs']

    # Drop rows with missing values
    data_df = data_df.dropna()

    # Remove outliers using Z-score 
    numeric_columns = [v for v in data_df.select_dtypes(include=['float64', 'int64']).columns if v != PARAMS['ml']['labels']]
    for column in numeric_columns:
        values = (data_df[column] - data_df[column].mean()).abs() / data_df[column].std() - PARAMS['ml']['outliers_std_factor']
        data_df = data_df[values < PARAMS['ml']['tolerance']]

    # label encoding
    label_encoder = LabelEncoder()
    string_columns = [v for v in data_df.select_dtypes(exclude=['float64', 'int64']).columns if v != PARAMS['ml']['labels']]
    for v in string_columns:
        data_df[v + ENCODED_SUFFIX] = label_encoder.fit_transform(data_df[v])

    return {
        'dfs': [
            {'df': data_df, 
             'table_name': TABLE_NAMES['clean_data']
             }]
        }

"""
normalization related functions; if normalization algorithm is changed, only these functions must change
"""

def normalize(df):
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    scaler.fit(df)
    return scaler

def normalize_column(df, column: str):
    """
    normalize df[column]

    return tuple that can be directly inserted into the normalization table
    """

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    df[column] = scaler.fit_transform(df[column].values.reshape(-1, 1))
    return (column, scaler.data_min_[0], scaler.data_max_[0], scaler.scale_[0], scaler.min_[0])

def normalization_transform_column(df, column: str, values: dict):
    # must be equivalent to MinMaxScaler.transform
    df[column] = (df[column] - values['min'])/values['scale']

    return df

def denormalize_columns(df, normalization_values):
    """
    denormalize columns in df; based on the code in MinMaxScaler in sklearn

    column is a column name in df; normalization is the dataframe with normalization values

    must be equivalant to MinMaxScaler.inverse_transform
    """

    for column in df.columns:
        values = normalization_values[normalization_values['name'] == column]
        if values.empty and column != PARAMS['ml']['labels']:
            print('Column {column} not found in the normalization data table.')
        else:
            values = values.iloc[0].to_dict()
            df[column] = df[column] * values['scale'] + values['min']

    return df

"""
end of normalization functions
"""

@from_table_to_df(TABLE_NAMES['clean_data'], None)
def normalize_data_func(**kwargs):
    """
    normalization
    split to train/test
    """
    
    from sklearn.model_selection import train_test_split
    import pandas as pd

    df = kwargs['dfs']

    # Split the data into training and test sets
    df_train, df_test = train_test_split(df, test_size=PARAMS['ml']['train_test_ratio'], random_state=42)

    # Normalize numerical columns
    normalization_values = [] # 
    for column in [v for v in df_train.select_dtypes(include=['float64', 'int64']).columns if v != PARAMS['ml']['labels']]:
        normalization_values.append(normalize_column(df_train, column))
    
    normalization_df = pd.DataFrame(data=normalization_values, columns=NORMALIZATION_TABLE_COLUMN_NAMES)

    return {
        'dfs': [
            {
                'df': df_train, 
                'table_name': TABLE_NAMES['train_data']
            },
            {
                'df': df_test, 
                'table_name': TABLE_NAMES['test_data']
            },
            {
                'df': normalization_df, 
                'table_name': TABLE_NAMES['normalization_data']
            }]
        }

@from_table_to_df(TABLE_NAMES['train_data'], None)
def train_model_func(**kwargs):
    """
    train model on train data
    """
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    import pickle

    train_df = kwargs['dfs']

    X_train = train_df.drop(columns=[PARAMS['ml']['labels']])
    y_train = train_df[PARAMS['ml']['labels']]

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save the model
    with open(PARAMS['files']['model'], 'wb') as f:
        pickle.dump(model, f)

    return {'status': 1}

@from_table_to_df(TABLE_NAMES['test_data'], None)
def predict_func(**kwargs):
    """
    predict on test data; denormalize
    """

    import pandas as pd
    import pickle

    test_df = kwargs['dfs']

    # Load the model
    with open(PARAMS['files']['model'], 'rb') as f:
        model = pickle.load(f)

    X_test = test_df.drop(columns=[PARAMS['ml']['labels']])
    y_test = test_df[PARAMS['ml']['labels']]

    # Predict
    predictions = model.predict(X_test)

    result_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': predictions
    })

    return {
        'dfs': [
            {
                'df': result_df, 
                'table_name': TABLE_NAMES['result']
            }]
        }

with DAG(
    'wine_data_pipeline',
    default_args=default_args,
    description='A simple wine data pipeline',
    schedule_interval=None,
) as dag:

    start = DummyOperator(task_id='start')

    get_data = S3ToLocalOperator(
        task_id='get_data',
        bucket_name=PARAMS['s3']['bucket'],
        filename=PARAMS['s3']['key'],
        dest=PARAMS['files']['local_file']
    )

    add_data_to_table = PythonOperator(
        task_id='add_data_to_table',
        python_callable=add_data_to_table_func
    )

    clean_data = PythonOperator(
        task_id='clean_data',
        python_callable=clean_data_func
    )

    normalize_data = PythonOperator(
        task_id='normalize_data',
        python_callable=normalize_data_func
    )

    train_model = PythonOperator(
        task_id='train_model',
        python_callable=train_model_func
    )

    predict = PythonOperator(
        task_id='predict',
        python_callable=predict_func
    )

    end = DummyOperator(task_id='end')

    start >> get_data >> add_data_to_table >> clean_data >> normalize_data >> train_model >> predict >> end
