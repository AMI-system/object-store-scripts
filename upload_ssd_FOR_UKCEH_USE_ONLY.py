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
from aiohttp import BasicAuth, ClientTimeout, FormData
from tenacity import retry, wait_fixed, stop_after_attempt
import tqdm.asyncio

# Replace these with actual credentials
USERNAME = "aimappuser"
PASSWORD = "Osd7r0I9hkFWLY0Eqoia"

async def upload_files_in_batches(name, bucket, dep_id, data_type, files, batch_size=100):
    """Upload files in batches with a second check after the first upload."""
    async with aiohttp.ClientSession(timeout=ClientTimeout(total=1200)) as session:
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

    async with session.post(url, auth=BasicAuth(USERNAME, PASSWORD), data=data) as response:
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

    await asyncio.gather(*tasks)

@retry(wait=wait_fixed(2), stop=stop_after_attempt(5))
async def get_presigned_url(session, name, bucket, dep_id, data_type, file_name, file_type):
    """Get a presigned URL for file upload."""
    url = "https://connect-apps.ceh.ac.uk/ami-data-upload/generate-presigned-url/"

    data = FormData()
    data.add_field("name", name)
    data.add_field("country", bucket)
    data.add_field("deployment", dep_id)
    data.add_field("data_type", data_type)
    data.add_field("filename", file_name)
    data.add_field("file_type", file_type)

    async with session.post(url, auth=BasicAuth(USERNAME, PASSWORD), data=data) as response:
        response.raise_for_status()
        return await response.json()

@retry(wait=wait_fixed(2), stop=stop_after_attempt(5))
async def upload_file_to_s3(session, presigned_url, file_path, file_type):
    """Upload file content to S3 using the presigned URL."""
    headers = {"Content-Type": file_type}
    try:
        with open(file_path, "rb") as file:
            data = file.read()
            async with session.put(presigned_url, data=data, headers=headers) as response:
                response.raise_for_status()
                await response.text()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def get_file_info(file_path):
    """Get file information including name, content, and type."""
    filename = os.path.basename(file_path)
    file_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
    return filename, file_type

def gather_files(ssd_path):
    """Gather files from the specified folders on the SSD, excluding $RECYCLE.BIN."""
    def filter_recycle_bin(file_list):
        """Exclude files in the $RECYCLE.BIN folder."""
        return [file for file in file_list if "$RECYCLE.BIN" not in str(file) and ".Trashes" not in str(file)]

    paths = {
        "snapshot_images": filter_recycle_bin(list(pathlib.Path(ssd_path, "images").rglob("*.jpg"))),
        "audible_recordings": filter_recycle_bin(list(pathlib.Path(ssd_path, "audio").rglob("*.wav"))),
        "ultrasound_recordings": filter_recycle_bin(list(pathlib.Path(ssd_path, "ultrasonic").rglob("*.wav")))
    }
    
    for data_type, files in paths.items():
        print(f"Found {len(files)} files in {data_type}.")
    
    return paths

# Specify bucket, dep_id, and the SSD path containing data folders
bucket = "gbr"
dep_id = "dep000065"
ssd_path = "E:/"

# Run the upload process
if __name__ == "__main__":
    fullname = "Dylan Carbone"
    data_paths = gather_files(ssd_path)
    
    async def main():
        for data_type, files in data_paths.items():
            if files:
                await upload_files_in_batches(fullname, bucket, dep_id, data_type, files)
            else:
                print(f"No files found in {data_type} folder. Skipping.")

    # Run all tasks in a single event loop
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())    
    asyncio.run(main())