"""Query list of all contributors and reviewers in a release"""

from sh.contrib import git
import sys
import re
import requests
import json

if len(sys.argv) != 5:
    print(f'Usage: {sys.argv[0]} [starting commit/tag] [ending commit/tag] [GitHub username] [GitHub password]')
    sys.exit(1)

from_commit = sys.argv[1]
to_commit = sys.argv[2]
username = sys.argv[3]
password = sys.argv[4]

contributors = set()
reviewers = set()

for line in git.log(f'{from_commit}..{to_commit}', '--pretty=format:%s', '--reverse'):
    m = re.search('\(#([0-9]+)\)', line.rstrip())
    if m:
        pr_id = m.group(1)
        print(f'PR #{pr_id}')

        r = requests.get(f'https://api.github.com/repos/dmlc/xgboost/pulls/{pr_id}/commits', auth=(username, password))
        assert r.status_code == requests.codes.ok, f'Code: {r.status_code}, Text: {r.text}'
        commit_list = json.loads(r.text)
        try:
            contributors.update([commit['author']['login'] for commit in commit_list])
        except TypeError:
            contributors.update(str(input(f'Error fetching contributors for PR #{pr_id}. Enter it manually, as a space-separated list:')).split(' '))

        r = requests.get(f'https://api.github.com/repos/dmlc/xgboost/pulls/{pr_id}/reviews', auth=(username, password))
        assert r.status_code == requests.codes.ok, f'Code: {r.status_code}, Text: {r.text}'
        review_list = json.loads(r.text)
        reviewers.update([x['user']['login'] for x in review_list])

        r = requests.get(f'https://api.github.com/repos/dmlc/xgboost/issues/{pr_id}/comments', auth=(username, password))
        assert r.status_code == requests.codes.ok, f'Code: {r.status_code}, Text: {r.text}'
        comment_list = json.loads(r.text)
        reviewers.update([x['user']['login'] for x in comment_list])

print('Contributors:', end='')
for x in sorted(contributors):
    r = requests.get(f'https://api.github.com/users/{x}', auth=(username, password))
    assert r.status_code == requests.codes.ok, f'Code: {r.status_code}, Text: {r.text}'
    user_info = json.loads(r.text)
    if user_info['name'] is None:
        print(f"@{x}, ", end='')
    else:
        print(f"{user_info['name']} (@{x}), ", end='')

print('Reviewers:', end='')
for x in sorted(reviewers):
    r = requests.get(f'https://api.github.com/users/{x}', auth=(username, password))
    assert r.status_code == requests.codes.ok, f'Code: {r.status_code}, Text: {r.text}'
    user_info = json.loads(r.text)
    if user_info['name'] is None:
        print(f"@{x}, ", end='')
    else:
        print(f"{user_info['name']} (@{x}), ", end='')
