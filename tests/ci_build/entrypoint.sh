#!/usr/bin/env bash

# This script is a wrapper creating the same user inside container as the one
# running the ci_build.sh outside the container. It also set the home directory
# for the user inside container to match the same absolute path as the workspace
# outside of container.  Do not run this manually. It does not make sense. It is
# intended to be called by ci_build.sh only.

set -e

COMMAND=("$@")

if ! touch /this_is_writable_file_system; then
  echo "You can't write to your filesystem!"
  echo "If you are in Docker you should check you do not have too many images" \
      "with too many files in them. Docker has some issue with it."
  exit 1
else
  rm /this_is_writable_file_system
fi

if [[ -n $CI_BUILD_UID ]] && [[ -n $CI_BUILD_GID ]]; then
    groupadd -o -g "${CI_BUILD_GID}" "${CI_BUILD_GROUP}"
    useradd -o -m -g "${CI_BUILD_GID}" -u "${CI_BUILD_UID}" \
        "${CI_BUILD_USER}"
    export HOME="/home/${CI_BUILD_USER}"
    shopt -s dotglob
    cp -r /root/* "$HOME/"
    chown -R "${CI_BUILD_UID}:${CI_BUILD_GID}" "$HOME"

    # Allows project-specific customization
    if [[ -e "/workspace/.pre_entry.sh" ]]; then
        gosu "${CI_BUILD_UID}:${CI_BUILD_GID}" /workspace/.pre_entry.sh
    fi

    # Enable passwordless sudo capabilities for the user
    chown root:"${CI_BUILD_GID}" "$(which gosu)"
    chmod +s "$(which gosu)"; sync

    exec gosu "${CI_BUILD_UID}:${CI_BUILD_GID}" "${COMMAND[@]}"
else
    exec "${COMMAND[@]}"
fi
