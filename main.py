import functions_framework
from load_to_bq import *
from functions import *

# Triggered by a change in a storage bucket
@functions_framework.cloud_event
def dataUpload(cloud_event):
    
    # cloud_event contains the event (upload of feature_store.csv in GCS) which triggers this Cloud Function
    data = cloud_event.data

    # Metadata of cloud_event which can be used to pass info to functions
    event_id = cloud_event["id"]
    event_type = cloud_event["type"]
    bucket = data["bucket"]
    metageneration = data["metageneration"]
    timeCreated = data["timeCreated"]
    updated = data["updated"]

    # The metadata that we need; contains name of uploaded file in GCS
    name = data["name"]

    # Checks to prevent ghost-triggers
    if name.startswith('feature'):
        filename = str(name)
        print(f"Filename: {filename}")
        createStore(filename)	# calls the main function of load_to_bq.py
    else:
        print('No file starting from feature') 



    