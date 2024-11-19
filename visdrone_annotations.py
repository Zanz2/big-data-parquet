import pandas as pd
import glob
from pathlib import Path

def read_visdrone_annotations(annotations_dir):
    """
    Read VisDrone annotations into a DataFrame.
    
    VisDrone format: <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
    Object categories:
    0: ignored regions
    1: pedestrian
    2: people
    3: bicycle
    4: car
    5: van
    6: truck
    7: tricycle
    8: awning-tricycle
    9: bus
    10: motor
    11: others
    """
    
    # Define column names for the annotations
    columns = [
        'bbox_left', 'bbox_top', 'bbox_width', 'bbox_height',
        'score', 'category_id', 'truncation', 'occlusion',
        'image_name'  # Will be added from filename
    ]
    
    # Define category mapping
    category_map = {
        0: 'ignored regions',
        1: 'pedestrian',
        2: 'people',
        3: 'bicycle',
        4: 'car',
        5: 'van',
        6: 'truck',
        7: 'tricycle',
        8: 'awning-tricycle',
        9: 'bus',
        10: 'motor',
        11: 'others'
    }
    
    all_annotations = []
    
    # Read all annotation files
    for ann_file in Path(annotations_dir).glob('*.txt'):
        image_name = ann_file.stem + '.jpg'  # Assuming images are JPG
        
        # Read annotations
        try:
            # Read the file, handling empty files
            df = pd.read_csv(ann_file, header=None)
            if not df.empty:
                #print(ann_file)
                #print(df.columns)
                #print(df.columns[:-1])
                df.columns = columns[:-1]  # Exclude image_name from columns
                df['image_name'] = image_name
                all_annotations.append(df)
        except pd.errors.EmptyDataError:
            print(f"Empty annotation file: {ann_file}")
            continue
    
    # Combine all annotations
    if all_annotations:
        final_df = pd.concat(all_annotations, ignore_index=True)
        
        # Add category names
        final_df['category_name'] = final_df['category_id'].map(category_map)
        
        # Calculate additional bbox coordinates
        final_df['bbox_right'] = final_df['bbox_left'] + final_df['bbox_width']
        final_df['bbox_bottom'] = final_df['bbox_top'] + final_df['bbox_height']
        
        return final_df
    else:
        return pd.DataFrame(columns=columns)


if __name__ == "__main__":
    # Example usage
    type_map_list = ["test-dev","val","train"]

    for type_map in type_map_list:
        print(f"Writing parquet {type_map}")
        input_path = f"E:\\public_datasets\\VisDrone\\{type_map}\\annotations"
        output_path = f"E:\\public_datasets\\VisDrone_parquet\\annotations\\{type_map}"
        
        df = read_visdrone_annotations(input_path)
        print("\nDataset Statistics:")
        print(f"Total annotations: {len(df)}")
        print("\nAnnotations per category:")
        print(df['category_name'].value_counts())
        df.to_parquet(f"{output_path}\\annotations.parquet") # no compression and extra settings because the files are tiny
        print("Success")


    