#!/bin/bash

whl_name=$1
dest_file=$2

ver_commit=`echo $1 | awk -F- '{print $2}'`
ver=`echo $ver_commit | awk -F+ '{print $1}'`
commit_id=`echo $ver_commit | awk -F+ '{print $2}'`

cat << EOF > $dest_file
<?xml version="1.0" encoding="UTF-8"?>
<root>
  <version>${ver}</version>
  <commit>${commit_id}</commit>
</root>
EOF

