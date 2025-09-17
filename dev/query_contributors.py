"""Query list of all contributors and reviewers in a release"""

import json
import re
import sys

import requests
from sh.contrib import git

if len(sys.argv) != 5:
    print(f'Usage: {sys.argv[0]} [starting commit/tag] [ending commit/tag] [GitHub username] ' +
           '[GitHub password]')
    sys.exit(1)

from_commit = sys.argv[1]
to_commit = sys.argv[2]
username = sys.argv[3]
password = sys.argv[4]

contributors = set()
reviewers = set()

def paginate_request(url, callback):
    r = requests.get(url, auth=(username, password))
    assert r.status_code == requests.codes.ok, f'Code: {r.status_code}, Text: {r.text}'
    callback(json.loads(r.text))
    while 'next' in r.links:
        r = requests.get(r.links['next']['url'], auth=(username, password))
        callback(json.loads(r.text))

for line in git.log(f'{from_commit}..{to_commit}', '--pretty=format:%s', '--reverse', '--first-parent'):
    m = re.search('\(#([0-9]+)\)$', line.rstrip())
    if m:
        pr_id = m.group(1)
        print(f'PR #{pr_id}')

        def process_commit_list(commit_list):
            try:
                contributors.update([commit['author']['login'] for commit in commit_list])
            except TypeError:
                prompt = (f'Error fetching contributors for PR #{pr_id}. Enter it manually, ' +
                          'as a space-separated list: ')
                contributors.update(str(input(prompt)).split(' '))
        def process_review_list(review_list):
            reviewers.update([x['user']['login'] for x in review_list])
        def process_comment_list(comment_list):
            reviewers.update([x['user']['login'] for x in comment_list])

        paginate_request(f'https://api.github.com/repos/dmlc/xgboost/pulls/{pr_id}/commits',
                         process_commit_list)
        paginate_request(f'https://api.github.com/repos/dmlc/xgboost/pulls/{pr_id}/reviews',
                         process_review_list)
        paginate_request(f'https://api.github.com/repos/dmlc/xgboost/issues/{pr_id}/comments',
                         process_comment_list)

print('Contributors: ', end='')
for x in sorted(contributors):
    r = requests.get(f'https://api.github.com/users/{x}', auth=(username, password))
    assert r.status_code == requests.codes.ok, f'Code: {r.status_code}, Text: {r.text}'
    user_info = json.loads(r.text)
    if user_info['name'] is None:
        print(f"@{x}, ", end='')
    else:
        print(f"{user_info['name']} (@{x}), ", end='')

print('\nReviewers: ', end='')
for x in sorted(reviewers):
    r = requests.get(f'https://api.github.com/users/{x}', auth=(username, password))
    assert r.status_code == requests.codes.ok, f'Code: {r.status_code}, Text: {r.text}'
    user_info = json.loads(r.text)
    if user_info['name'] is None:
        print(f"@{x}, ", end='')
    else:
        print(f"{user_info['name']} (@{x}), ", end='')
print('')
