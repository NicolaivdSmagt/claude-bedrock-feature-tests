#!/usr/bin/env python3
import boto3, io, os, sys
from PIL import Image

# Add parent dirs to path so we can import load_config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from load_config import load_config, get_bedrock_client

# AWS credentials are set via environment variables (e.g. AWS_PROFILE=work)

# Create test image
buffer = io.BytesIO()
Image.new("RGB", (50, 50), "blue").save(buffer, format="JPEG")
image_bytes = buffer.getvalue()

config = load_config()
client = get_bedrock_client(config)
model_id = config["bedrock_model_id"]

for count in [20, 21]:
    content = [{"text": f"Count {count} images"}] + [
        {"image": {"format": "jpeg", "source": {"bytes": image_bytes}}}
        for _ in range(count)
    ]

    print(
        f"\nTesting {count} images... (Content: {len(content)} items, Image: {len(image_bytes)} bytes)"
    )

    try:
        response = client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": content}],
            inferenceConfig={"maxTokens": 20},
            additionalModelRequestFields={"anthropic_beta": ["context-1m-2025-08-07"]},
        )
        print(f":white_check_mark: {count} images: SUCCESS")
        print(
            f"Response: {response.get('output', {}).get('message', {}).get('content', [])}"
        )
    except Exception as e:
        print(f":x: {count} images: FAILED - {type(e).__name__}: {e}")
        if hasattr(e, "response"):
            print(f"Error response: {e.response}")
        print(f"Model ID: {model_id}")
