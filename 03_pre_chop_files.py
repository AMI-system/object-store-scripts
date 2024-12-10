import json
import argparse
from math import ceil
import os

def load_workload(input_file, file_extensions):
    """
    Load workload from a file. Assumes each line contains an S3 key.
    """
    with open(input_file, "r", encoding='UTF-8') as f:
        all_keys = [line.strip() for line in f.readlines()]

    subset_keys = [x for x in all_keys if x.endswith(tuple(file_extensions))]

    # remove corrupt keys
    subset_keys = [x for x in subset_keys if not os.path.basename(x).startswith("$")]

    # remove keys uploaded from the recycle bin (legacy code)
    subset_keys = [x for x in subset_keys if "recycle" not in x]
    print(f"{len(subset_keys)} keys")
    
    return subset_keys

def split_workload(keys, chunk_size):
    """
    Split a list of keys into chunks of a specified size.
    """
    num_chunks = ceil(len(keys) / chunk_size)
    chunks = {
        str(i + 1): {"keys": keys[i * chunk_size: (i + 1) * chunk_size]}
        for i in range(num_chunks)
    }
    return chunks

def save_chunks(chunks, output_file):
    """
    Save chunks to a JSON file.
    """
    with open(output_file, 'w') as f:
        json.dump(chunks, f, indent=4)

def main():
    parser = argparse.ArgumentParser(
        description="Pre-chop S3 workload into manageable chunks."
    )
    parser.add_argument(
        "--input_file", 
        type=str, 
        required=True, 
        help="Path to file containing S3 keys, one per line."
    )
    parser.add_argument(
        "--file_extensions",
        type=str,
        nargs="+",
        required=True,
        default="'jpg' 'jpeg'",
        help="File extensions to be chuncked. If empty, all extensions used.",
    )
    parser.add_argument(
        "--chunk_size", 
        type=int, 
        default=100, 
        help="Number of keys per chunk."
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        required=True, 
        help="Path to save the output JSON file."
    )
    args = parser.parse_args()

    # Load the workload from the input file
    keys = load_workload(args.input_file, args.file_extensions)

    # Split the workload into chunks
    chunks = split_workload(keys, args.chunk_size)

    # Save the chunks to a JSON file
    save_chunks(chunks, args.output_file)

    print(f"Successfully split {len(keys)} keys into {len(chunks)} chunks.")
    print(f"Chunks saved to {args.output_file}")

if __name__ == "__main__":
    main()
