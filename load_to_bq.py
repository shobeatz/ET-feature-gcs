from google.cloud import bigquery, storage
import pandas as pd
from functions import *

def createStore(filename):
    
    # print(filename) 

    ##### TO-DO #####
    project_id = "your-gcp-project-I'd"
    dataset_id = "your-dataset"
    table_id = "feature_store"
    bucket_name = "your-bucket-name"

    # Creating BQ table reference
    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    
    # Creating BQ client
    client = bigquery.Client()

    # Reading file from cloud_event as filename which triggers this Cloud Function
    df = pd.read_csv(f"gs://{bucket_name}/"+filename) 
    
    # Changing timestamp to correct data type & format
    df['event_timestamp'] = pd.to_datetime(df['event_timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
    df['event_timestamp'] = df['event_timestamp'].astype('datetime64[ns]')

    # Creating a system entry time for BQ batch insert
    df["createdAt"] = pd.to_datetime('now').strftime('%Y-%m-%d %H:%M:%S')  
    df["createdAt"] = df["createdAt"].astype("datetime64[ns]")  

    # Dynamically get the new schema and new data
    new_schema = get_new_schema_from_dataframe(df)

    try:

        # Getting existing schema
        existing_schema = get_existing_schema(client, table_ref)

        # Identifying missing columns
        missing_columns = find_missing_columns(existing_schema, new_schema)

        # Adding missing columns to BQ
        add_missing_columns(client, table_ref, missing_columns)

        # Appending new column data into BQ
        append_new_column_data(client, table_ref, df, missing_columns)

    except: 

	# get_existing_schema will raise an error if there is no BQ table, hence creating BQ table
        job_config = bigquery.LoadJobConfig(schema=new_schema, write_disposition="WRITE_TRUNCATE")

        # Convert DataFrame to BigQuery table
        job = client.load_table_from_dataframe(df, f"{project_id}.{dataset_id}.{table_id}", job_config=job_config)
        job.result()  

        print("Table created with explicit schema!")



