import os
import pandas as pd

def combine_csvs_from_subdirectories(root_directory):
    """
    Reads all CSV files from the subdirectories of the root directory and combines them into a single DataFrame.
    
    Args:
        root_directory (str): Path to the root directory containing subdirectories with CSV files.
    
    Returns:
        pd.DataFrame: Combined DataFrame containing data from all CSV files.
    """
    all_csv_files = []
    combined_df = pd.DataFrame()
    # Walk through the root directory and its subdirectories
    for subdir, _, files in os.walk(root_directory):
        files = [x for x in files if x.endswith('.csv')]
        for file in files:
            full_path = os.path.join(subdir, file)
            all_csv_files.append(full_path)

    all_csv_files = [x for x in all_csv_files if 'checkpoint' not in x and 'combined' not in x]
    
    for path in all_csv_files:
        print(path)
        temp = pd.read_csv(path, on_bad_lines='skip', dtype='unicode', header=None)
        temp = temp.loc[temp[13].notna(), ]
        combined_df = pd.concat([
            combined_df, temp            
        ], ignore_index=True)
        
    
    return combined_df

# Example usage
if __name__ == "__main__":
    root_dir = "/home/users/katriona/object-store-scripts/data/harlequin/"  # Replace with your root directory path
    combined_dataframe = combine_csvs_from_subdirectories(root_dir)
    #print(combined_dataframe.head())  # Display the first few rows of the combined DataFrame
    print("complete")

    # Optional: Save the combined DataFrame to a new CSV
    combined_dataframe.to_csv("/home/users/katriona/object-store-scripts/data/harlequin/combined_data.csv", index=False)