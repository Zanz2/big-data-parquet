import pandas as pd
from pathlib import Path
import json
import numpy as np
import os
from PIL import Image
import io

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
    
    
    # TODO for images: iterate recursively over all the folders and all the contents and make a dataframe with byte images and filenames then convert this to a parquet file
    # iter_save_parque_chunks(output_parquet_path+"annotations/", annotations_df, chunk_size=10000000)


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
        
def image_to_bytes(image_path):
    """Convert image to raw bytes without JPG container"""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # Save as raw bytes without container
            byte_arr = io.BytesIO()
            img.save(byte_arr, format='RAW')
            return byte_arr.getvalue()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None
    
    
claude_vomit = """

def process_images_in_chunks(root_dir, chunk_size=4000, output_dir='chunks_output'):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all jpg files recursively
    jpg_files = glob.glob(os.path.join(root_dir, '**/*.jpg'), recursive=True)
    
    # Process in chunks
    for chunk_idx, i in enumerate(range(0, len(jpg_files), chunk_size)):
        chunk_files = jpg_files[i:i + chunk_size]
        
        # Process chunk
        data = {
            'filename': [],
            'image_bytes': []
        }
        
        for file_path in chunk_files:
            image_bytes = image_to_bytes(file_path)
            if image_bytes is not None:
                data['filename'].append(file_path)
                data['image_bytes'].append(image_bytes)
        
        # Create DataFrame and save to parquet
        if data['filename']:  # Only save if there's data
            df = pd.DataFrame(data)
            output_file = os.path.join(output_dir, f'chunk_{chunk_idx}.parquet')
            df.to_parquet(
                output_file,
                compression='snappy',
                index=False
            )
            print(f"Saved chunk {chunk_idx} with {len(df)} images to {output_file}")
"""

if __name__ == "__main__":
    # Example usage
    input_path = "/media/zanz/backup_disk_work/public_datasets/objects365/train/zhiyuan_objv2_train.json"
    output_path = "/home/zanz/work_github/big-data-parquet/output/objects365_parquet/annotations/train/"
    
    convert_objects365_to_parquet(
        input_json_path=input_path,
        output_parquet_path=output_path,
        batch_size=4000
    )
