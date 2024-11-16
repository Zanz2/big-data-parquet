import pandas as pd
from pathlib import Path
import numpy as np
import os
from PIL import Image
import cv2

def convert_dataset_to_parquet(input_image_folder: str, output_parquet_folder: str, batch_size: int):

    # Create output directory if it doesn't exist
    output_dir = Path(output_parquet_folder).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    full_paths = []
    filenames = []
    chunk_index = 0
    
    for root, _, files in os.walk(input_image_folder):
        for file in files:
            if file.lower().endswith('.jpg'):
                full_path = os.path.abspath(os.path.join(root, file))
                full_paths.append(full_path)
                filenames.append(file)
            
            if len(filenames) == batch_size:
                assert len(filenames) == len(full_paths)
                print("Writing parquet")
                processed_chunk_df = process_images(full_paths, filenames)
                iter_save_parque_chunks(output_parquet_folder, processed_chunk_df, chunk_index ,chunk_index*batch_size)
                chunk_index += 1
                full_paths = []
                filenames = []
            elif len(filenames) > batch_size:
                raise Exception("Error, too many files in batch, check code")
    
    print("Writing parquet")
    processed_chunk_df = process_images(full_paths, filenames)
    iter_save_parque_chunks(output_parquet_folder, processed_chunk_df, chunk_index ,chunk_index*batch_size)


def iter_save_parque_chunks(output_dir, df, index, image_start):
    output_dir_parent = Path(output_dir)
    output_dir_parent.mkdir(parents=True, exist_ok=True)
    output_file = os.path.join(output_dir, f'chunk_{index}_{image_start}.parquet')
    df.to_parquet(
        output_file,
        compression='snappy',
        index=False
    )
    print(f"Saved chunk {index} to {output_file}")
        
    

#---------------------------------------------------------
    
def process_images(filepaths_list, filenames_list):
    images_data = []
    for index, file_path in enumerate(filepaths_list):
        img_info = {}
        try:
            # Get file extension
            img_format = file_path.split('.')[-1].lower()
            
            # Read image as bytes directly
            with open(file_path, 'rb') as f:
                img_bytes = f.read()
            
            # Get image info without keeping the full image in memory
            with Image.open(file_path) as img:
                img_info['filename'] = filenames_list[index]
                img_info['image_bytes'] = img_bytes
                img_info['format'] = img_format
                img_info['width'] = img.width
                img_info['height'] = img.height
                img_info['mode'] = img.mode
                images_data.append(img_info)
                
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue
    print(images_data[0]["filename"])
    print(images_data[0]["format"])
    print(images_data[0]["width"])
    print(images_data[0]["height"])
    print(images_data[0]["mode"])
    return pd.DataFrame(images_data)



def parquet_image_to_cv2(image_data: bytearray, image_shape: np.ndarray):
    img = np.frombuffer(image_data, dtype=np.uint8)
    img = img.reshape((image_shape))
    im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return im_rgb
        

def s3file_to_cv2(s3file):
    arr = np.asarray(bytearray(s3file), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    return img


if __name__ == "__main__":
    # Example usage
    input_path = "/media/zanz/backup_disk_work/public_datasets/bdd100k/bdd100k/bdd100k/images/100k/val/"
    #output_path = "/home/zanz/work_github/big-data-parquet/output/dataset_parquet/data/val/"
    output_path = "/media/zanz/backup_disk_work/public_datasets/bdd100k_parquet/data/val/"
    batch_size = 11000
    
    convert_dataset_to_parquet(
        input_image_folder=input_path,
        output_parquet_folder=output_path,
        batch_size=batch_size
    )
    
    input_path = "/media/zanz/backup_disk_work/public_datasets/bdd100k/bdd100k/bdd100k/images/100k/test/"

    output_path = "/media/zanz/backup_disk_work/public_datasets/bdd100k_parquet/data/test/"
    convert_dataset_to_parquet(
        input_image_folder=input_path,
        output_parquet_folder=output_path,
        batch_size=batch_size
    )

    input_path = "/media/zanz/backup_disk_work/public_datasets/bdd100k/bdd100k/bdd100k/images/100k/train/"

    output_path = "/media/zanz/backup_disk_work/public_datasets/bdd100k_parquet/data/train/"
    convert_dataset_to_parquet(
        input_image_folder=input_path,
        output_parquet_folder=output_path,
        batch_size=batch_size
    )
