# This file will be used by load_to_bq.py 

from google.cloud import bigquery
import pandas as pd

def get_new_schema_from_dataframe(df):
    try:
        """Infers schema from a Pandas DataFrame."""
        dtype_mapping = {
            "int64": "INT64",
            "float64": "FLOAT64",
            "object": "STRING",
            "bool": "BOOL",
            "datetime64[ns]": "TIMESTAMP",
        }
        return [{"name": col, "type": dtype_mapping.get(str(df[col].dtype), "STRING")} for col in df.columns]
    except Exception as e:
        print("Error in: get_new_schema_from_dataframe()")

def get_existing_schema(client, table_ref):
    try:
        """Fetches the existing schema from the BigQuery table."""
        table = client.get_table(table_ref)
        return {field.name: field for field in table.schema}  # Dictionary for quick lookup
    except Exception as e:
        print(f"Error in: get_existing_schema() so creating new table.")
        raise

def find_missing_columns(existing_schema, new_schema):
    try:
        """Identifies columns that are in the new schema but not in the existing schema."""
        existing_column_names = set(existing_schema.keys())
        new_columns = [col for col in new_schema if col["name"] not in existing_column_names]
        return new_columns
    except Exception as e:
        print("Error in: find_missing_columns()")

def add_missing_columns(client, table_ref, missing_columns):
    try:
        """Adds new columns to the BigQuery table."""
        if not missing_columns:
            print("No new columns to add.")
            return

        table = client.get_table(table_ref)
        updated_schema = table.schema + [bigquery.SchemaField(col["name"], col["type"]) for col in missing_columns]
        table.schema = updated_schema
        client.update_table(table, ["schema"])

        print(f"Added new columns: {[col['name'] for col in missing_columns]}")
    except Exception as e:
        print("Error in: add_missing_columns()")
        
def append_new_column_data(client, table_ref, df, missing_columns):
    try:
        """Appends new column data to BigQuery, ensuring old columns get NULL values."""
        if not missing_columns:
            print("No new columns to append.")
            return

        # Fetch full schema after update
        table = client.get_table(table_ref)
        all_columns = [field.name for field in table.schema]

        # Ensure DataFrame has all columns (fill missing ones with NULL)
        for col in all_columns:
            if col not in df.columns:
                df[col] = None  # Add missing columns with NULL values

        # Reorder columns to match BigQuery schema order
        df = df[all_columns]

        # Append data to BigQuery
        job = client.load_table_from_dataframe(df, table_ref)
        job.result()  # Wait for job to finish
        print(f"Appended {len(df)} new rows into {table_ref}")
    except Exception as e:
        print("Error in: append_new_column_data()")