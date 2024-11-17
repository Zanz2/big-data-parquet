import lmdb
import pyarrow as pa
import pyarrow.parquet as pq
import os
from pathlib import Path

def convert_lsun_to_parquet_chunks(mdb_dir, output_dir, chunk_size=11000):
    """
    Convert LSUN .mdb files to multiple smaller parquet files
    
    Args:
        mdb_dir (str): Directory containing data.mdb and lock.mdb files
        output_dir (str): Output directory for parquet files
        chunk_size (int): Number of images per chunk
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create schema
    schema = pa.schema([
        ('key', pa.string()),
        ('image', pa.binary())
    ])
    
    # Open LMDB environment
    env = lmdb.open(mdb_dir, readonly=True, lock=False)
    
    try:
        with env.begin() as txn:
            cursor = txn.cursor()
            batch_data = {'key': [], 'image': []}
            file_counter = 0
            
            for idx, (key, value) in enumerate(cursor):
                # Add data to current batch
                batch_data['key'].append(key.decode('utf-8'))
                batch_data['image'].append(value)
                
                # Write batch when chunk_size is reached
                if (idx + 1) % chunk_size == 0:
                    output_file = os.path.join(output_dir, f'part_{file_counter:04d}.parquet')
                    table = pa.Table.from_pydict(batch_data, schema=schema)
                    
                    # Write the chunk to a new parquet file
                    pq.write_table(
                        table, 
                        output_file,
                        compression='snappy',
                        row_group_size=chunk_size
                    )
                    
                    print(f"Written file {output_file} with {chunk_size} records")
                    file_counter += 1
                    batch_data = {'key': [], 'image': []}
            
            # Write remaining data if any
            if batch_data['key']:
                output_file = os.path.join(output_dir, f'part_{file_counter:04d}.parquet')
                table = pa.Table.from_pydict(batch_data, schema=schema)
                pq.write_table(
                    table, 
                    output_file,
                    compression='snappy',
                    row_group_size=len(batch_data['key'])
                )
                print(f"Written final file {output_file} with {len(batch_data['key'])} records")
    
    finally:
        env.close()
        print(f"Conversion complete. Files saved in {output_dir}")

if __name__ == "__main__":
    mdb_dir = "E:\\public_datasets\\lsun\\motorbike"  # Directory containing data.mdb and lock.mdb
    output_path = "E:\\public_datasets\\lsun_parquet\\data\\motorbike"
    convert_lsun_to_parquet_chunks(mdb_dir, output_path)

    mdb_dir = "E:\\public_datasets\\lsun\\car"  # Directory containing data.mdb and lock.mdb
    output_path = "E:\\public_datasets\\lsun_parquet\\data\\car"
    convert_lsun_to_parquet_chunks(mdb_dir, output_path)
