#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script provides an interactive console application for uploading files to an S3 bucket using presigned URLs.
It guides the user through selecting the deployment and data type and then uploads the specified files.
"""

import sys
import os
import mimetypes
import pathlib
import getpass
import asyncio
import nest_asyncio
import requests
from requests.auth import HTTPBasicAuth
import aiohttp
from aiohttp import BasicAuth, ClientTimeout, FormData
from tenacity import retry, wait_fixed, stop_after_attempt
import tqdm.asyncio

# Apply nested asyncio to allow event loops to run inside Jupyter notebooks or other nested environments.
nest_asyncio.apply()


# Global variables to store credentials
GLOBAL_USERNAME = None
GLOBAL_PASSWORD = None


def clear_screen():
    """Clear the terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")


def get_input(prompt):
    """Get user input."""
    return input(prompt + ": ")


def get_choice(prompt, options):
    """Get user's choice from a list of options."""
    print(prompt)
    for i, option in enumerate(options, start=1):
        print(f"{i}. {option}")
    while True:
        choice = input("Choose an option (enter the number): ")
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        print("Invalid choice. Please try again.")


def display_menu():
    """Display the main menu and handle user interaction."""
    clear_screen()
    print("Upload Files")
    print("============\n")

    global GLOBAL_USERNAME, GLOBAL_PASSWORD

    if GLOBAL_USERNAME is None or GLOBAL_PASSWORD is None:
        GLOBAL_USERNAME = get_input("API Username")
        GLOBAL_PASSWORD = getpass.getpass("API Password: ")

    all_deployments = get_deployments()

    fullname = get_input("\nYour Full Name")

    countries = list({d["country"] for d in all_deployments if d["status"] == "active"})
    countries.append("Test")
    country = get_choice("Countries:", countries)

    deployment = ""
    if country != "Test":
        country_deployments = [
            f"{d['location_name']} - {d['camera_id']}"
            for d in all_deployments
            if d["country"] == country and d["status"] == "active"
        ]
        deployment = get_choice("\nDeployments:", country_deployments)

    data_types = ["snapshot_images", "audible_recordings", "ultrasound_recordings"]
    data_type = get_choice("\nData type:", data_types)
    extension = ".jpg" if data_type == "snapshot_images" else ".wav"

    print("\nSelect Directory:")
    while True:
        directory_path = get_input("Enter directory path")
        if os.path.isdir(directory_path):
            break
        print("Invalid directory. Please try again.")

    files = [
        os.path.join(directory_path, f)
        for f in os.listdir(directory_path)
        if os.path.isfile(os.path.join(directory_path, f))
    ]
    directory_path = pathlib.Path(directory_path)
    files = list(directory_path.rglob(f"*{extension}"))

    print("\nReview Your Input")
    print("=================\n")
    print(f"Full Name: {fullname}")
    print(f"Country: {country}")
    print(f"Deployment: {deployment}")
    print(f"Data Type: {data_type}")
    print(f"Number of files: {len(files)}")

    confirm = get_input("\nUpload files? (yes/no)")
    if confirm.lower() == "yes":
        # print("\nUploading files...")
        if country != "Test":
            s3_bucket_name = [
                d["country_code"]
                for d in all_deployments
                if d["country"] == country and d["status"] == "active"
            ][0].lower()
            location_name, camera_id = deployment.split(" - ")
            dep_id = [
                d["deployment_id"]
                for d in all_deployments
                if d["country"] == country
                and d["location_name"] == location_name
                and d["camera_id"] == camera_id
                and d["status"] == "active"
            ][0]
        else:
            s3_bucket_name = "test-upload"
            dep_id = "dep_test"

        asyncio.run(
            upload_files_in_batches(fullname, s3_bucket_name, dep_id, data_type, files)
        )

        # print("Files uploaded successfully!")
        prompt_next_action()
    else:
        print("\nUpload canceled.")


def prompt_next_action():
    """Prompt the user for the next action: upload more files or leave."""
    while True:
        next_action = get_choice(
            "\nWhat do you want to do next?", ["Upload more files", "Leave"]
        )
        if next_action == "Upload more files":
            display_menu()
        elif next_action == "Leave":
            print("Goodbye!")
            sys.exit(0)
        else:
            print("Invalid choice. Please try again.")


def get_file_info(file_path):
    """Get file information including name, content, and type."""
    filename = os.path.basename(file_path)
    file_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
    return filename, file_type


def get_deployments():
    """Fetch deployments from the API with authentication."""
    try:
        url = "https://connect-apps.ceh.ac.uk/ami-data-upload/get-deployments/"
        response = requests.get(
            url, auth=HTTPBasicAuth(GLOBAL_USERNAME, GLOBAL_PASSWORD), timeout=600
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as err:
        print(f"HTTP Error: {err}")
        if response.status_code == 401:
            print("Wrong username or password. Try again!")
        sys.exit(1)
    except Exception as err:
        print(f"Error: {err}")
        sys.exit(1)


async def upload_files_in_batches(
    name, bucket, dep_id, data_type, files, batch_size=100
):
    """Upload files in batches."""
    async with aiohttp.ClientSession(timeout=ClientTimeout(total=1200)) as session:
        while True:
            print()
            progress_exist = tqdm.asyncio.tqdm(
                total=len(files), desc="Checking if files already in server"
            )
            files_to_upload = await check_files(
                session, name, bucket, dep_id, data_type, files, progress_exist
            )
            progress_exist.close()
            print(
                f"{len(files_to_upload)} files missing from the server. The upload will start in a few moments..."
            )
            print()

            if not files_to_upload:
                print("All files have been uploaded successfully.")
                break

            progress_bar = tqdm.asyncio.tqdm(
                total=len(files_to_upload), desc="Uploading files"
            )

            if len(files_to_upload) <= batch_size:
                await upload_files(
                    session, name, bucket, dep_id, data_type, files_to_upload
                )
                progress_bar.update(len(files_to_upload))
            else:
                for i in range(0, len(files_to_upload), batch_size):
                    end = i + batch_size
                    batch = files_to_upload[i:end]
                    await upload_files(session, name, bucket, dep_id, data_type, batch)
                    progress_bar.update(len(batch))

            progress_bar.close()

            # Update files list to only include those that still need to be checked
            files = files_to_upload


async def check_files(session, name, bucket, dep_id, data_type, files, progress_exist):
    """Check if files exists in the object store already."""
    files_to_upload = []

    for file_path in files:
        if not await check_file_exist(
            session, name, bucket, dep_id, data_type, file_path
        ):
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

    async with session.post(
        url, auth=BasicAuth(GLOBAL_USERNAME, GLOBAL_PASSWORD), data=data
    ) as response:
        response.raise_for_status()
        exist = await response.json()
        return exist["exists"]


async def upload_files(session, name, bucket, dep_id, data_type, files):
    """Upload individual files."""
    tasks = []
    for file_path in files:
        file_name, file_type = get_file_info(file_path)
        try:
            presigned_url = await get_presigned_url(
                session, name, bucket, dep_id, data_type, file_name, file_type
            )
            task = upload_file_to_s3(session, presigned_url, file_path, file_type)
            tasks.append(task)
        except Exception as e:
            print(f"Error getting presigned URL for {file_name}: {e}")

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

    async with session.post(
        url, auth=BasicAuth(GLOBAL_USERNAME, GLOBAL_PASSWORD), data=data
    ) as response:
        response.raise_for_status()
        return await response.json()


@retry(wait=wait_fixed(2), stop=stop_after_attempt(5))
async def upload_file_to_s3(session, presigned_url, file_path, file_type):
    """Upload file content to S3 using the presigned URL."""
    headers = {"Content-Type": file_type}
    try:
        with open(file_path, "rb") as file:
            data = file.read()
            async with session.put(
                presigned_url, data=data, headers=headers
            ) as response:
                response.raise_for_status()
                await response.text()
    except aiohttp.ClientError:
        pass
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    display_menu()
