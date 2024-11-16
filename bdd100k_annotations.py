import pandas as pd
from pathlib import Path
import json
import numpy as np
import os
import ast

def convert_dataset_to_parquet(input_json_path: str, output_parquet_path: str, batch_size: int = 10000):
    """
    Convert dataset annotations to Parquet format
    
    Args:
        input_json_path: Path to dataset JSON annotation file
        output_parquet_path: Path to save the Parquet file
        batch_size: Number of annotations to process at once
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_parquet_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(input_json_path) as f:
        data = json.load(f)
        # Choose the key you want to convert to DataFrame

    #>>> data[0].keys()
    #dict_keys(['name', 'attributes', 'timestamp', 'labels'])

    #>>> data[0]["name"]
    #'0000f77c-6257be58.jpg'

    #>>> data[0]["attributes"]
    #{'weather': 'clear', 'scene': 'city street', 'timeofday': 'daytime'}

    #>>> data[0]["timestamp"]
    #10000

    #>>> data[0]["labels"]
    #type(list)

    dataframe_entity = pd.DataFrame(data)
    #dataframe_entity.join(pd.DataFrame(dataframe_entity.pop('attributes').values.tolist()))

    df_attributes = pd.json_normalize(dataframe_entity['attributes'])
    dataframe_entity = pd.concat([dataframe_entity.drop('attributes', axis=1), df_attributes], axis=1)



    #dataframe_entity['attributes'] = dataframe_entity['attributes'].apply()
    #all_length = None
    #for index, row in dataframe_entity.iterrows():
    #    #print(row["attributes"])
    #    print(row)
    #    break
    #    if all_length is None:
    #        all_length = len(row["attributes"])
    #    assert all_length == len(row["attributes"])
    print(dataframe_entity.columns)
    dataframe_entity.to_parquet(
            output_parquet_path+"annotations.parquet",
            compression='snappy',
            index=False
        )
    

if __name__ == "__main__":
    # Example usage
    input_path = "/media/zanz/backup_disk_work/public_datasets/bdd100k/bdd100k/bdd100k/labels/bdd100k_labels_images_train.json"
    output_path = "/media/zanz/backup_disk_work/public_datasets/bdd100k_parquet/annotations/train/"
    
    convert_dataset_to_parquet(
        input_json_path=input_path,
        output_parquet_path=output_path,
        batch_size=4000
    )
