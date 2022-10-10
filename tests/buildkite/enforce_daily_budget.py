import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--response", type=str, required=True)
    args = parser.parse_args()
    with open(args.response, "r") as f:
        payload = f.read()
    response = json.loads(payload)
    if response["approved"]:
        print(f"Testing approved. Reason: {response['reason']}")
    else:
        raise RuntimeError(f"Testing rejected. Reason: {response['reason']}")
