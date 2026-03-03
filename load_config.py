# ABOUTME: Shared config loader for all test scripts in this directory.
# ABOUTME: Reads config.yaml and provides helpers for Bedrock/Anthropic client setup.

import json
import os

import boto3
import yaml


def load_config():
    """Load configuration from config.yaml in the same directory as this module."""
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config.yaml"
    )
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_bedrock_client(config=None):
    """Create a Bedrock Runtime client using config values."""
    if config is None:
        config = load_config()
    return boto3.client("bedrock-runtime", region_name=config["region"])


def get_anthropic_api_key(config=None):
    """Fetch Anthropic API key from AWS Secrets Manager using config values."""
    if config is None:
        config = load_config()

    sm_region = config["secrets_manager_region"]
    sm_secret = config["secrets_manager_secret_name"]

    client = boto3.client(service_name="secretsmanager", region_name=sm_region)

    try:
        response = client.get_secret_value(SecretId=sm_secret)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch API key from Secrets Manager: {e}")

    secret = response["SecretString"]

    try:
        secret_dict = json.loads(secret)
        if "api_key" in secret_dict:
            return secret_dict["api_key"]
        elif "ANTHROPIC_API_KEY" in secret_dict:
            return secret_dict["ANTHROPIC_API_KEY"]
        elif "key" in secret_dict:
            return secret_dict["key"]
        else:
            return next(iter(secret_dict.values()))
    except json.JSONDecodeError:
        return secret.strip()


def get_anthropic_client(config=None):
    """Create an Anthropic API client using config values and Secrets Manager."""
    import anthropic

    if config is None:
        config = load_config()
    api_key = get_anthropic_api_key(config)
    return anthropic.Anthropic(api_key=api_key)
