#!/usr/bin/env python3
# ABOUTME: Tests maximum image size limits on AWS Bedrock (single and multiple images).
# ABOUTME: Uses Converse API with 1M context window for multi-image tests.

import os
import json
import sys
import base64
import boto3
from datetime import datetime
from pathlib import Path

# Add parent dirs to path so we can import load_config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from load_config import load_config, get_bedrock_client


def encode_image(image_path):
    """Read and base64 encode an image file"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_file_size_mb(file_path):
    """Get file size in megabytes"""
    return os.path.getsize(file_path) / (1024 * 1024)


def test_image_bedrock(client, image_path, model_id):
    """Test sending a single image to AWS Bedrock"""
    file_size = get_file_size_mb(image_path)

    try:
        image_data = encode_image(image_path)

        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": "What do you see in this image? Please provide a brief description.",
                        },
                    ],
                }
            ],
        }

        response = client.invoke_model(modelId=model_id, body=json.dumps(request_body))

        response_body = json.loads(response["body"].read())

        return {
            "file": os.path.basename(image_path),
            "size_mb": file_size,
            "success": True,
            "message_id": response_body.get("id", "N/A"),
            "input_tokens": response_body.get("usage", {}).get("input_tokens", 0),
            "output_tokens": response_body.get("usage", {}).get("output_tokens", 0),
        }

    except Exception as e:
        return {
            "file": os.path.basename(image_path),
            "size_mb": file_size,
            "success": False,
            "error_type": type(e).__name__,
            "error_message": str(e),
        }


def test_multiple_images_bedrock(client, image_path, count, model_id):
    """Test sending multiple copies of the same image to AWS Bedrock with 1M context window"""
    file_size = get_file_size_mb(image_path)
    total_size = file_size * count

    try:
        image_data = encode_image(image_path)
        encoded_size_mb = len(image_data) / (1024 * 1024)
        total_encoded_mb = encoded_size_mb * count

        # Build content array for Converse API
        content = []
        for i in range(count):
            content.append(
                {
                    "image": {
                        "format": "jpeg",
                        "source": {"bytes": base64.b64decode(image_data)},
                    }
                }
            )

        content.append(
            {
                "text": f"I've sent you {count} copies of the same image. Can you confirm you can see all {count} images?"
            }
        )

        # Use Converse API with 1M context window
        response = client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": content}],
            inferenceConfig={"maxTokens": 1024},
            additionalModelRequestFields={"anthropic_beta": ["context-1m-2025-08-07"]},
        )

        return {
            "test": f"{count}x {os.path.basename(image_path)}",
            "count": count,
            "size_per_image_mb": file_size,
            "total_raw_mb": total_size,
            "encoded_per_image_mb": encoded_size_mb,
            "total_encoded_mb": total_encoded_mb,
            "success": True,
            "message_id": response.get("responseMetadata", {}).get("requestId", "N/A"),
            "input_tokens": response.get("usage", {}).get("inputTokens", 0),
            "output_tokens": response.get("usage", {}).get("outputTokens", 0),
        }

    except Exception as e:
        return {
            "test": f"{count}x {os.path.basename(image_path)}",
            "count": count,
            "size_per_image_mb": file_size,
            "total_raw_mb": total_size,
            "encoded_per_image_mb": len(image_data) / (1024 * 1024)
            if "image_data" in locals()
            else 0,
            "total_encoded_mb": (len(image_data) / (1024 * 1024) * count)
            if "image_data" in locals()
            else 0,
            "success": False,
            "error_type": type(e).__name__,
            "error_message": str(e),
        }


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def main():
    config = load_config()
    model_id = config["bedrock_model_id"]
    region = config["region"]

    print("=" * 80)
    print("IMAGE SIZE LIMIT TEST: AWS Bedrock")
    print(f"Model: {model_id}")
    print(f"Region: {region}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 80)

    files_dir = Path(os.path.dirname(os.path.abspath(__file__)), "..", "..", "files")
    image_files = sorted(files_dir.glob("*.jpg"))

    if not image_files:
        print(f"\nNo image files found in {files_dir}/")
        return

    print(f"\nFound {len(image_files)} image(s) to test:")
    for img in image_files:
        size_mb = get_file_size_mb(img)
        print(f"  - {img.name} ({size_mb:.2f} MB)")

    client = get_bedrock_client(config)

    # Single image tests
    print_section("BEDROCK - Single Image Tests")

    results = []
    for image_path in image_files:
        file_size = get_file_size_mb(image_path)
        print(
            f"\nTesting: {os.path.basename(image_path)} ({file_size:.2f} MB)... ",
            end="",
            flush=True,
        )
        result = test_image_bedrock(client, image_path, model_id)
        results.append(result)

        if result["success"]:
            print(f"SUCCESS")
        else:
            print(f"FAILED ({result['error_type']})")

    print_section("BEDROCK - Single Image Summary")
    for result in results:
        status = "OK" if result["success"] else "FAIL"
        print(f"{status} {result['file']} ({result['size_mb']:.2f} MB)")
        if result["success"]:
            print(
                f"   Input: {result['input_tokens']} tokens, Output: {result['output_tokens']} tokens"
            )
        else:
            print(f"   Error: {result['error_type']}")

    successful_sizes = [r["size_mb"] for r in results if r["success"]]
    failed_sizes = [r["size_mb"] for r in results if not r["success"]]

    if successful_sizes and failed_sizes:
        print(
            f"\nImage size limit is between {max(successful_sizes):.2f} MB and {min(failed_sizes):.2f} MB"
        )
    elif successful_sizes:
        print(
            f"\nAll images up to {max(successful_sizes):.2f} MB were successfully processed"
        )

    # Multiple image tests
    print_section("BEDROCK - Multiple Image Tests (with 1M context window)")

    three_mb_image = files_dir / "3mb.jpg"
    if three_mb_image.exists():
        test_counts = [6, 5]
        multi_results = []
        for count in test_counts:
            print(f"\nTest: Trying {count}x copies of 3mb.jpg... ", end="", flush=True)
            result = test_multiple_images_bedrock(
                client, three_mb_image, count, model_id
            )
            multi_results.append(result)
            print(
                f"SUCCESS" if result["success"] else f"FAILED ({result['error_type']})"
            )

        print(f"\nMultiple Image Summary:")
        for result in multi_results:
            status = "OK" if result["success"] else "FAIL"
            print(
                f"\n{status} {result['count']} images - {result['total_encoded_mb']:.2f} MB encoded"
            )
            if result["success"]:
                print(f"   Input: {result['input_tokens']} tokens")
            else:
                print(f"   Error: {result['error_type']}")
                print(f"   Message: {result['error_message'][:200]}...")
    else:
        print("\n3mb.jpg not found, skipping multiple image test")

    print("\nAll tests completed")


if __name__ == "__main__":
    main()
