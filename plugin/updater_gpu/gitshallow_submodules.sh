#!/bin/bash
git submodule init
for i in $(git submodule | awk '{print $2}'); do
    spath=$(git config -f .gitmodules --get submodule.$i.path)
    surl=$(git config -f .gitmodules --get submodule.$i.url)
    if [ $spath == "cub" ]
    then
        git submodule update --depth 3 $spath
    else
        git submodule update $spath
    fi
done
