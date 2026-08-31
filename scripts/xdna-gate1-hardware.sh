#!/usr/bin/env bash

# Run the real-device Gate-1 harness in the current shell.  This wrapper only
# selects project-local executable and user-supplied external artifacts; it
# never changes resource limits or invokes a system management utility.
set -u
export LC_ALL=C

script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
gate1_bin=${XDNA_GATE1_BIN:-"$project_dir/build/xdna-gate1"}
xclbin=${XDNA_GATE1_XCLBIN:-}
elf=${XDNA_GATE1_ELF:-}

if [[ -z "$xclbin" ]]; then
    printf 'XDNA_GATE1_XCLBIN must name an extracted or standalone xclbin\n' >&2
    printf 'This wrapper does not materialize or copy protected artifacts.\n' >&2
    exit 2
fi
if [[ ! -x "$gate1_bin" ]]; then
    printf 'Gate-1 executable not found: %s\n' "$gate1_bin" >&2
    exit 2
fi

args=(--xclbin "$xclbin")
if [[ -n "$elf" ]]; then
    args+=(--elf "$elf")
fi
if [[ "$#" -gt 0 ]]; then
    args+=("$@")
fi

exec "$gate1_bin" "${args[@]}"
