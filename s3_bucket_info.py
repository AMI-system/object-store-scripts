#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script counts the total number of files in an S3 bucket for a specific
country, deployment, and data type. AWS credentials and S3 bucket name are
loaded from a configuration file (credentials.json).
"""

import sys
import os
import getpass
import json
from yaspin import yaspin
import boto3
import requests
from requests.auth import HTTPBasicAuth


# Load AWS credentials and S3 bucket name from config file
with open('credentials.json') as config_file:
    aws_credentials = json.load(config_file)

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=aws_credentials['AWS_ACCESS_KEY_ID'],
    aws_secret_access_key=aws_credentials['AWS_SECRET_ACCESS_KEY'],
    endpoint_url=aws_credentials['AWS_URL_ENDPOINT'],
    region_name=aws_credentials['AWS_REGION']
)


def get_deployments(username, password):
    """Fetch deployments from the API with authentication."""
    try:
        url = "https://connect-apps.ceh.ac.uk/ami-data-upload/get-deployments/"
        response = requests.get(url, auth=HTTPBasicAuth(username, password))
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


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


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
        else:
            print("Invalid choice. Please try again.")


@yaspin(text="Calculating...")
def number_of_files(s3_bucket_name, prefix):
    """
    Count number of files for country, deployment and data type.
    """
    # Create a paginator helper for list_objects_v2
    paginator = s3_client.get_paginator('list_objects_v2')

    # Define the parameters for the pagination operation
    operation_parameters = {
        'Bucket': s3_bucket_name,
        'Prefix': prefix
    }

    # Create an iterator for the paginated response
    page_iterator = paginator.paginate(**operation_parameters)

    # Initialize a counter for the total number of files
    count = 0

    # Iterate through each page in the paginated response
    for page in page_iterator:
        # Add the number of keys in the current page to the total count
        count += page['KeyCount']

    return count


def display_menu():
    """Display the main menu and handle user interaction."""
    clear_screen()
    print("Upload Files")
    print("============\n")

    username = get_input("API Username")
    password = getpass.getpass("API Password: ")

    all_deployments = get_deployments(username, password)

    countries = list(set([d["country"] for d in all_deployments if d["status"] == "active"]))
    country = get_choice("Countries:", countries)

    country_deployments = [
        f"{d['location_name']} - {d['camera_id']}" for d in all_deployments
        if d["country"] == country and d["status"] == "active"
    ]
    deployment = get_choice("\nDeployments:", country_deployments)

    data_types = ["snapshot_images", "audible_recordings", "ultrasound_recordings"]
    data_type = get_choice("\nData type:", data_types)

    s3_bucket_name = [d["country_code"] for d in all_deployments if d["country"] == country and d["status"] == "active"][0].lower()
    location_name, camera_id = deployment.split(" - ")
    dep_id = [d["deployment_id"] for d in all_deployments if d["country"] == country and d["location_name"] == location_name and d["camera_id"] == camera_id and d["status"] == "active"][0]

    prefix = f"{dep_id}/{data_type}"

    count = number_of_files(s3_bucket_name, prefix)

    # Print the total number of files for the specified parameters
    print(
        f"\nTotal number of files for {country}, deployment {deployment} and data type {data_type}: {count}\n"
    )


if __name__ == '__main__':
    display_menu()
