#!/bin/bash
ls -alLR ${CACHE_PREFIX}
# Avoid growing the cache.
brew cleanup
