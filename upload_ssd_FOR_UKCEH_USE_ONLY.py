#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script uploads files from structured folders on an SSD to an S3 bucket using presigned URLs.
The bucket, deployment ID, and the SSD path are specified directly in the script.
"""

import os
import mimetypes
import asyncio
import pathlib
import aiohttp
from aiohttp import BasicAuth, ClientTimeout, FormData, ClientResponseError
from tenacity import retry, wait_fixed, stop_after_attempt
import tqdm.asyncio

# Replace these with actual credentials
GLOBAL_USERNAME = "aimappuser".strip()
GLOBAL_PASSWORD = "Osd7r0I9hkFWLY0Eqoia".strip()

async def upload_files_in_batches(name, bucket, dep_id, data_type, files, batch_size=100):
    """Upload files in batches with a second check after the first upload."""
    async with aiohttp.ClientSession(timeout=ClientTimeout(total=3600)) as session:
        while True:
            print(f"\nChecking for files to upload for data type: {data_type}")
            progress_exist = tqdm.asyncio.tqdm(total=len(files), desc="Checking if files already in server")
            files_to_upload = await check_files(session, name, bucket, dep_id, data_type, files, progress_exist)
            progress_exist.close()
            
            if not files_to_upload:
                print(f"All files in {data_type} have been uploaded successfully.")
                break

            print(f"\n{len(files_to_upload)} {data_type} files missing from the server. Starting upload...")
            progress_bar = tqdm.asyncio.tqdm(total=len(files_to_upload), desc="Uploading files")

            for i in range(0, len(files_to_upload), batch_size):
                end = i + batch_size
                batch = files_to_upload[i:end]
                await upload_files(session, name, bucket, dep_id, data_type, batch)
                progress_bar.update(len(batch))

            progress_bar.close()

            # Perform a second check to confirm successful upload
            print("\nPerforming a second check for any remaining files...")
            files = files_to_upload  # Re-check the files that were missing in the first attempt

async def check_files(session, name, bucket, dep_id, data_type, files, progress_exist):
    """Check if files exists in the object store already."""
    files_to_upload = []

    for file_path in files:
        if not await check_file_exist(session, name, bucket, dep_id, data_type, file_path):
            files_to_upload.append(file_path)
        progress_exist.update(1)

    return files_to_upload

async def check_file_exist(session, name, bucket, dep_id, data_type, file_path):
    """Check if files exists in the object store already."""
    url = "https://connect-apps.ceh.ac.uk/ami-data-upload/check-file-exist/"

    file_name, _ = get_file_info(file_path)
    data = FormData()
    data.add_field("name", name)
    data.add_field("country", bucket)
    data.add_field("deployment", dep_id)
    data.add_field("data_type", data_type)
    data.add_field("filename", file_name)
    data.add_field("username", GLOBAL_USERNAME)  # ← add this
    data.add_field("password", GLOBAL_PASSWORD)  # ← and this
    
    async with session.post(url, data=data) as response:
        response.raise_for_status()
        exist = await response.json()
        return exist["exists"]

async def upload_files(session, name, bucket, dep_id, data_type, files):
    """Upload individual files."""
    tasks = []
    for file_path in files:
        file_name, file_type = get_file_info(file_path)
        try:
            presigned_url = await get_presigned_url(session, name, bucket, dep_id, data_type, file_name, file_type)
            task = upload_file_to_s3(session, presigned_url, file_path, file_type)
            tasks.append(task)
        except Exception as e:
            print(f"Error getting presigned URL for {file_name}: {e}")

    # Directly gather tasks without additional error handling at this level
    await asyncio.gather(*tasks)

@retry(wait=wait_fixed(2), stop=stop_after_attempt(5))
async def get_presigned_url(
    session, name, bucket, dep_id, data_type, file_name, file_type
):
    """Get a presigned URL for file upload."""
    url = "https://connect-apps.ceh.ac.uk/ami-data-upload/generate-presigned-url/"

    data = FormData()
    data.add_field("name", name)
    data.add_field("country", bucket)
    data.add_field("deployment", dep_id)
    data.add_field("data_type", data_type)
    data.add_field("filename", file_name)
    data.add_field("file_type", file_type)
    data.add_field("username", GLOBAL_USERNAME)
    data.add_field("password", GLOBAL_PASSWORD)

    async with session.post(
        url, data=data
    ) as response:
        response.raise_for_status()
        return await response.json()

@retry(wait=wait_fixed(2), stop=stop_after_attempt(5))
async def upload_file_to_s3(session, presigned_url, file_path, file_type):
    headers = {"Content-Type": file_type}
    try:
        with open(file_path, "rb") as file:
            async with session.put(presigned_url, data=file, headers=headers) as response:
                response.raise_for_status()  # Raise exception for HTTP errors
                await response.text()
    except ClientResponseError as e:
        if e.status == 504:
            print("The object store service is taking too long to respond to the API. Please continue the upload.")
        elif e.status == 408:
            print("Your upload speed is slow. Please continue the upload.")
        else:
            print(f"HTTP Error {e.status}: {e.message}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def get_file_info(file_path):
    """Get file information including name and type."""
    if not isinstance(file_path, (str, bytes)):
        file_path = str(file_path)  # Convert PosixPath to string
    
    filename = os.path.basename(file_path)
    file_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
    return filename, file_type

def gather_files(ssd_path):
    """Gather files from the specified folders on the SSD, excluding small files, $RECYCLE.BIN, and corrupted files."""
    def filter_recycle_bin_and_small_files(file_list):
        """Exclude files in the $RECYCLE.BIN folder and files smaller than 10 KB."""
        valid_files = []
        for file in file_list:
            try:
                if "$RECYCLE.BIN" not in str(file) and ".Trashes" not in str(file) and ".Trash-0" not in str(file):
                    if os.path.getsize(file) >= 10 * 1024:  # File size >= 10 KB
                        valid_files.append(file)
            except OSError as e:
                print(f"Skipping file '{file}' due to error: {e}")
        return valid_files

    paths = {
        "snapshot_images": filter_recycle_bin_and_small_files(list(pathlib.Path(ssd_path).rglob("*.jpg"))),
        "audible_recordings": filter_recycle_bin_and_small_files(list(pathlib.Path(ssd_path, "audio").rglob("*.wav"))),
        "ultrasound_recordings": filter_recycle_bin_and_small_files(list(pathlib.Path(ssd_path, "ultrasonic").rglob("*.wav")))
    }
    
    for data_type, files in paths.items():
        print(f"Found {len(files)} valid files in {data_type}.")
    
    return paths

# Run the upload process
if __name__ == "__main__":

    # Specify your name, bucket, dep_id, and the SSD path containing data folders
    fullname = "Dylan Carbone"
    bucket = "gbr" # if you want to trial an upload to the test bucket, use 'test-upload'
    dep_id = "dep000058" # if you want to trial an upload to the test bucket, use 'dep_test'
    ssd_path = r"D:\teams agzero data\Wiltshire - Manor farm"

    # # Specify your name, bucket, dep_id, and the SSD path containing data folders
    # fullname = "Dylan Carbone"
    # bucket = "aia" # if you want to trial an upload to the test bucket, use 'test-upload'
    # dep_id = "dep000099" # if you want to trial an upload to the test bucket, use 'dep_test'
    # ssd_path = "E:/"
    
    # Define adaptive batch sizes
    batch_sizes = {
        "snapshot_images": 30,
        "audible_recordings": 20,
        "ultrasound_recordings": 10
    }

    data_paths = gather_files(ssd_path)
    
    async def main():
        for data_type, files in data_paths.items():
            if files:
                # Get the batch size for the current data type
                batch_size = batch_sizes.get(data_type)
                await upload_files_in_batches(fullname, bucket, dep_id, data_type, files, batch_size)
            else:
                print(f"No files found in {data_type} folder. Skipping.")

    # Run all tasks in a single event loop
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
