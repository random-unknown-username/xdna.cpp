#!/usr/bin/env bash
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(CDPATH= cd -- "$script_dir/.." && pwd -P)
temp_dir=$(mktemp -d /tmp/xdna-q4-0-reference.XXXXXX)
binary="$temp_dir/test-q4-0-reference"

cleanup() {
    rm -f -- "$binary"
    rmdir -- "$temp_dir"
}
trap cleanup EXIT

# Exact standalone compile command for the Q4_0 reference header and sources:
c++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -Werror \
    -I"$repo_root/include" \
    "$repo_root/src/xdna-q4-0-reference.cpp" \
    "$repo_root/tests/test-q4-0-reference.cpp" \
    -o "$binary"

"$binary"
