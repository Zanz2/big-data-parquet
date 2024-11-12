import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import json

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
    
    def process_batch(annotations):
        """Process a batch of annotations into a pandas DataFrame"""
        processed = []
        for ann in annotations:
            # Flatten the annotation structure
            record = {
                'image_id': ann.get('image_id'),
                'file_name': ann.get('file_name'),
                'width': ann.get('width'),
                'height': ann.get('height')
            }
            
            # Add annotation fields if they exist
            if 'annotations' in ann:
                for idx, a in enumerate(ann['annotations']):
                    record.update({
                        f'bbox_{idx}': str(a.get('bbox', [])),
                        f'category_id_{idx}': a.get('category_id'),
                        f'area_{idx}': a.get('area'),
                        f'iscrowd_{idx}': a.get('iscrowd', 0)
                    })
            processed.append(record)
        return pd.DataFrame(processed)

    # Read and process annotations in batches
    current_batch = []
    first_batch = True
    
    print(f"Converting {input_json_path} to {output_parquet_path}")
    
    with open(input_json_path, 'r') as f:
        for line in f:
            ann = json.loads(line.strip())
            current_batch.append(ann)
            
            if len(current_batch) >= batch_size:
                df = process_batch(current_batch)
                
                if first_batch:
                    # Write first batch with schema
                    table = pa.Table.from_pandas(df)
                    with pq.ParquetWriter(output_parquet_path, table.schema, compression='snappy') as writer:
                        writer.write_table(table)
                    first_batch = False
                else:
                    # Append subsequent batches
                    table = pa.Table.from_pandas(df)
                    with pq.ParquetWriter(output_parquet_path, table.schema, compression='snappy', append=True) as writer:
                        writer.write_table(table)

                    # TODO TEMP debug
                    stop_now = input("Would you like to stop? (y/n)")
                    if "y" in stop_now: return 
                
                current_batch = []
                print(f"Processed {batch_size} annotations...")
        
        # Process remaining annotations
        if current_batch:
            df = process_batch(current_batch)
            table = pa.Table.from_pandas(df)
            with pq.ParquetWriter(output_parquet_path, table.schema, compression='snappy', append=True) as writer:
                writer.write_table(table)

    print(f"Conversion completed: {output_parquet_path}")

if __name__ == "__main__":
    # Example usage
    input_path = "/mnt/backup_disk_work/public_datasets/objects365/val/zhiyuan_objv2_val.json"
    output_path = "/home/zanza/work_projects/dataset_to_parquet/output/parquet_val.parquet"
    
    convert_objects365_to_parquet(
        input_json_path=input_path,
        output_parquet_path=output_path,
        batch_size=3000
    )
