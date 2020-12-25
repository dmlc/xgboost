#!/usr/bin/env bash

# Helper script for creating release tarball.

print_usage() {
    printf "Script for making release source tarball.\n"
    printf "Usage:\n\trelease-tarball.sh <TAG>\n\n"
}

print_error() {
    local msg=$1
    printf "\u001b[31mError\u001b[0m: $msg\n\n"
    print_usage
}

check_input() {
    local TAG=$1
    if [ -z $TAG ]; then
        print_error "Empty tag argument"
        exit -1
    fi
}

check_curdir() {
    local CUR_ABS=$1
    printf "Current directory: ${CUR_ABS}\n"
    local CUR=$(basename $CUR_ABS)

    if [ $CUR == "dev" ]; then
        cd ..
        CUR=$(basename $(pwd))
    fi

    if [ $CUR != "xgboost" ]; then
        print_error "Must be in project root or xgboost/dev.  Current directory: ${CUR}"
        exit -1;
    fi
}

# Remove all submodules.
cleanup_git() {
    local TAG=$1
    check_input $TAG

    git checkout $TAG || exit -1

    local SUBMODULES=$(grep "path = " ./.gitmodules | cut -f 3 --delimiter=' ' -)

    for module in $SUBMODULES; do
        rm -rf ${module}/.git
    done

    rm -rf .git
}

make_tarball() {
    local SRCDIR=$1
    local CUR_ABS=$2
    tar -czf xgboost.tar.gz xgboost

    printf "Copying ${SRCDIR}/xgboost.tar.gz back to ${CUR_ABS}/xgboost.tar.gz .\n"
    cp xgboost.tar.gz ${CUR_ABS}/xgboost.tar.gz
    printf "Writing hash to ${CUR_ABS}/hash .\n"
    sha256sum -z ${CUR_ABS}/xgboost.tar.gz | cut -f 1 --delimiter=' ' > ${CUR_ABS}/hash
}

main() {
    local TAG=$1
    check_input $TAG

    local CUR_ABS=$(pwd)
    check_curdir $CUR_ABS

    local TMPDIR=$(mktemp -d)
    printf "tmpdir: ${TMPDIR}\n"

    git clean -xdf || exit -1
    cp -R . $TMPDIR/xgboost
    pushd .

    cd $TMPDIR/xgboost
    cleanup_git $TAG

    cd ..
    make_tarball $TMPDIR $CUR_ABS

    popd
    rm -rf $TMPDIR
}

main $1
