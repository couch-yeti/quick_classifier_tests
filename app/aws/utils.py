import sys
import boto3

platform = sys.platform
if "win" in platform:
    SESSION = boto3.Session(profile_name="dev")
else:
    SESSION = boto3
