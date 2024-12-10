import boto3
import argparse
import json


def list_s3_keys(bucket_name, deployment_id=""):
    """
    List all keys in an S3 bucket under a specific prefix.

    Parameters:
        bucket_name (str): The name of the S3 bucket.
        prefix (str): The prefix to filter keys (default: "").

    Returns:
        list: A list of S3 object keys.
    """
    with open("./credentials.json", encoding="utf-8") as config_file:
        aws_credentials = json.load(config_file)

    session = boto3.Session(
        aws_access_key_id=aws_credentials["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=aws_credentials["AWS_SECRET_ACCESS_KEY"],
        region_name=aws_credentials["AWS_REGION"],
    )
    s3_client = session.client("s3", endpoint_url=aws_credentials["AWS_URL_ENDPOINT"])

    keys = []
    continuation_token = None

    while True:
        list_kwargs = {
            "Bucket": bucket_name,
            "Prefix": deployment_id,
        }
        if continuation_token:
            list_kwargs["ContinuationToken"] = continuation_token

        response = s3_client.list_objects_v2(**list_kwargs)

        # Add object keys to the list
        for obj in response.get("Contents", []):
            keys.append(obj["Key"])

        # Check if there are more objects to list
        if response.get("IsTruncated"):  # If True, there are more results
            continuation_token = response["NextContinuationToken"]
        else:
            break

    return keys


def save_keys_to_file(keys, output_file):
    """
    Save S3 keys to a file, one per line.

    Parameters:
        keys (list): List of S3 keys.
        output_file (str): Path to the output file.
    """
    with open(output_file, "w") as f:
        for key in keys:
            f.write(key + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a file containing S3 keys from a bucket."
    )
    parser.add_argument(
        "--bucket", type=str, required=True, help="Name of the S3 bucket."
    )
    parser.add_argument(
        "--deployment_id",
        type=str,
        default="",
        help="The deployment id to filter objects. If set to '' then all deployments are used. (default: '')",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="s3_keys.txt",
        help="Output file to save S3 keys.",
    )
    args = parser.parse_args()

    # List keys from the specified S3 bucket and prefix
    print(
        f"Listing keys from bucket '{args.bucket}' with deployment '{args.deployment_id}'..."
    )
    keys = list_s3_keys(args.bucket, args.deployment_id)

    # Save keys to the output file
    save_keys_to_file(keys, args.output_file)
    print(f"Saved {len(keys)} keys to {args.output_file}")


if __name__ == "__main__":
    main()
