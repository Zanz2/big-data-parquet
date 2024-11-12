import pandas as pd
from pathlib import Path
import json
import numpy as np
import os

def convert_objects365_to_parquet(input_json_path: str, output_parquet_path: str, batch_size: int = 10000):
    """
    Convert Objects365 annotations to Parquet format
    
    Args:
        input_json_path: Path to Objects365 JSON annotation file
        output_parquet_path: Path to save the Parquet file
        batch_size: Number of annotations to process at once
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_parquet_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(input_json_path) as f:
        data = json.load(f)
        # Choose the key you want to convert to DataFrame
    
    images_df = pd.DataFrame(data['images'])
    images_df.to_parquet(
            output_parquet_path+"data_images.parquet",
            compression='snappy',
            index=False
        )
    
    annotations_df = pd.DataFrame(data['annotations'])
    annotations_df.to_parquet(
            output_parquet_path+"data_annotations.parquet",
            compression='snappy',
            index=False
        )
    
    categories_df = pd.DataFrame(data['categories'])
    categories_df.to_parquet(
            output_parquet_path+"data_categories.parquet",
            compression='snappy',
            index=False
        )
    
    licenses_df = pd.DataFrame(data['licenses'])
    licenses_df.to_parquet(
            output_parquet_path+"data_licenses.parquet",
            compression='snappy',
            index=False
        )
    

def iter_save_parque_chunks(output_dir, df, chunk_size):
    output_dir_parent = Path(output_dir)
    output_dir_parent.mkdir(parents=True, exist_ok=True)
    for i, chunk in enumerate(np.array_split(df, len(df) // chunk_size)):
        output_file = os.path.join(output_dir, f'chunk_{i}.parquet')
        chunk.to_parquet(
            output_file,
            compression='snappy',
            index=False
        )
        print(f"Saved chunk {i} to {output_file}")

if __name__ == "__main__":
    # Example usage
    input_path = "/media/zanz/backup_disk_work/public_datasets/objects365/train/zhiyuan_objv2_train.json"
    output_path = "/home/zanz/work_github/big-data-parquet/output/objects365_parquet/annotations/train/"
    
    convert_objects365_to_parquet(
        input_json_path=input_path,
        output_parquet_path=output_path,
        batch_size=4000
    )
