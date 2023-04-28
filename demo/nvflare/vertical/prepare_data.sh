#!/bin/bash

set -e

rm -fr ./*.pem /tmp/nvflare/poc

world_size=2

# Generate server and client certificates.
openssl req -x509 -newkey rsa:2048 -days 7 -nodes -keyout server-key.pem -out server-cert.pem -subj "/C=US/CN=localhost"
openssl req -x509 -newkey rsa:2048 -days 7 -nodes -keyout client-key.pem -out client-cert.pem -subj "/C=US/CN=localhost"

# Download HIGGS dataset.
if [ -f "HIGGS.csv" ]; then
  echo "HIGGS.csv exists, skipping download."
else
  echo "Downloading HIGGS dataset."
  wget https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz
  gunzip HIGGS.csv.gz
fi

# Split into train/test.
if [[ -f higgs.train.csv && -f higgs.test.csv ]]; then
  echo "higgs.train.csv and higgs.test.csv exist, skipping split."
else
  echo "Splitting HIGGS dataset into train/test."
  head -n 10450000 HIGGS.csv > higgs.train.csv
  tail -n 550000 HIGGS.csv > higgs.test.csv
fi

# Split train and test files by column to simulate a federated environment.
site_files=(higgs.{train,test}.csv-site-*)
if [ ${#site_files[@]} -eq $((world_size*2)) ]; then
  echo "Site files exist, skipping split."
else
  echo "Splitting train/test into site files."
  total_cols=28  # plus label
  cols=$((total_cols/world_size))
  echo "Columns per site: $cols"
  for (( site=1; site<=world_size; site++ )); do
    if (( site == 1 )); then
      start=$((cols*(site-1)+1))
    else
      start=$((cols*(site-1)+2))
    fi
    if (( site == world_size )); then
      end=$((total_cols+1))
    else
      end=$((cols*site+1))
    fi
    echo "Site $site, columns $start-$end"
    cut -d, -f${start}-${end} higgs.train.csv > higgs.train.csv-site-"${site}"
    cut -d, -f${start}-${end} higgs.test.csv > higgs.test.csv-site-"${site}"
  done
fi

nvflare poc -n 2 --prepare
mkdir -p /tmp/nvflare/poc/admin/transfer/vertical-xgboost
cp -fr config custom /tmp/nvflare/poc/admin/transfer/vertical-xgboost
cp server-*.pem client-cert.pem /tmp/nvflare/poc/server/
for (( site=1; site<=world_size; site++ )); do
  cp server-cert.pem client-*.pem /tmp/nvflare/poc/site-"${site}"/
  ln -s "${PWD}"/higgs.train.csv-site-"${site}" /tmp/nvflare/poc/site-"${site}"/higgs.train.csv
  ln -s "${PWD}"/higgs.test.csv-site-"${site}" /tmp/nvflare/poc/site-"${site}"/higgs.test.csv
done
