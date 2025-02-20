"""Helper script for triggering Read the docs build.

See `doc/contrib/docs.rst <https://xgboost.readthedocs.io/en/stable/contrib/docs.html>`__
for more info.

"""

import json
import os
import pprint
from http.client import responses as http_responses

import requests  # type: ignore


def trigger_build(token: str) -> None:
    """Trigger RTD build."""

    event_path = os.environ["GITHUB_EVENT_PATH"]
    with open(event_path, "r") as fd:
        event: dict = json.load(fd)

    if event.get("pull_request", None) is None:
        # refs/heads/branch-name
        branch = event["ref"].split("/")[-1]
    else:
        branch = event["pull_request"]["number"]

    if branch == "master":
        # Use the `latest` tag, otherwise RTD wouldn't update the rendered doc.
        branch = "latest"

    URL = f"https://readthedocs.org/api/v3/projects/xgboost/versions/{branch}/builds/"
    HEADERS = {"Authorization": f"token {token}"}
    response = requests.post(URL, headers=HEADERS)
    # 202 means the build is successfully triggered.
    if response.status_code != 202:
        status_text = http_responses[response.status_code]
        raise RuntimeError(
            "ReadTheDocs returned an unexpected response: "
            f"{response.status_code} {status_text}, reason: {response.reason}"
        )
    pprint.pprint(response.json(), indent=4)


def main() -> None:
    token = os.getenv("RTD_AUTH_TOKEN")
    # GA redacts the secret by default, but we should still be really careful to not log
    # (expose) the token in the CI.
    if token is None:
        raise RuntimeError(
            "The RTD_AUTH_TOKEN environment variable must be set to a valid auth token for the"
            "ReadTheDocs service."
        )
    if len(token) == 0:
        print("Document build is not triggered.")
        return

    if not isinstance(token, str) or len(token) != 40:
        raise ValueError(f"Invalid token.")

    trigger_build(token)


if __name__ == "__main__":
    main()
