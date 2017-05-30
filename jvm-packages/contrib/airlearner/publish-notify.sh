#!/usr/bin/env bash

author=`git --no-pager show -s --format='%an' ${TRAVIS_COMMIT}`
text="{\"text\":\"Airlearner version ${TRAVIS_TAG} has been <https://travis-ci.org/jq/xgboost/builds/${TRAVIS_BUILD_ID}|built> and published by ${author}!\"}"

echo $text
curl -s -X POST -H 'Content-type: application/json' --data "$text" ${AEROSOLVE_CHANNEL}
