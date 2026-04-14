#!/usr/bin/env python3
"""Test the FastAPI server locally."""

import sys
from pathlib import Path

# Add to path
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

print("Testing GET /...")
try:
    response = client.get("/")
    print(f"Status: {response.status_code}")
    print(f"Content length: {len(response.text)}")
    print(f"First 200 chars: {response.text[:200]}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\nTesting GET /status...")
try:
    response = client.get("/status")
    print(f"Status: {response.status_code}")
    print(f"JSON: {response.json()}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
