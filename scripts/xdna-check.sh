#!/usr/bin/env bash

# Read-only local health check.  In particular, this script never invokes
# sudo, prlimit, xrt-smi, or any other command that changes system state.
set -u
export LC_ALL=C

script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
build_dir=${XDNA_BUILD_DIR:-"$project_dir/build"}
xdna_info=${XDNA_INFO:-"$build_dir/xdna-info"}
xrt_root=${XILINX_XRT:-/opt/xilinx/xrt}

printf 'xdna.cpp read-only check\n'
printf 'Project: %s\n\n' "$project_dir"

printf 'Current shell memlock:\n'
printf '  ulimit -l: %s KB\n' "$(ulimit -l 2>&1)"
limits_line=$(awk '$1 == "Max" && $2 == "locked" && $3 == "memory" {print}' \
    /proc/self/limits 2>/dev/null || true)
if [[ -n "$limits_line" ]]; then
    printf '  /proc/self/limits: %s\n' "$limits_line"
else
    printf '  /proc/self/limits: unavailable\n'
fi
printf '\n'

xrt_smi=$(command -v xrt-smi 2>/dev/null || true)
if [[ -z "$xrt_smi" && -x "$xrt_root/bin/xrt-smi" ]]; then
    xrt_smi="$xrt_root/bin/xrt-smi"
fi

printf 'XRT/plugin availability:\n'
if [[ -n "$xrt_smi" ]]; then
    printf '  xrt-smi: available (%s)\n' "$xrt_smi"
else
    printf '  xrt-smi: unavailable\n'
fi

if command -v pkg-config >/dev/null 2>&1; then
    xrt_pkgconfig="$xrt_root/lib/pkgconfig"
    if [[ -n "${PKG_CONFIG_PATH-}" ]]; then
        xrt_pkgconfig="$xrt_pkgconfig:$PKG_CONFIG_PATH"
    fi
    if xrt_version=$(PKG_CONFIG_PATH="$xrt_pkgconfig" \
        pkg-config --modversion xrt 2>&1); then
        printf '  xrt.pc: available (%s)\n' "$xrt_version"
    else
        printf '  xrt.pc: unavailable (%s)\n' "$xrt_version"
    fi
else
    printf '  xrt.pc: unavailable (pkg-config not found)\n'
fi

if [[ -x "$xdna_info" ]]; then
    printf '  xdna-info XRT probe/plugin: available (%s)\n' "$xdna_info"
else
    printf '  xdna-info XRT probe/plugin: unavailable (%s)\n' "$xdna_info"
fi
printf '\n'

xdna_status=127
printf 'xdna-info --strict output:\n'
if [[ -x "$xdna_info" ]]; then
    "$xdna_info" --strict
    xdna_status=$?
else
    printf 'xdna-info executable not found: %s\n' "$xdna_info"
fi
printf 'xdna-info exit status: %d\n\n' "$xdna_status"

printf 'Elevated memlock test command (print-only; paste into the same interactive shell):\n'
printf '  sudo prlimit --pid $$ --memlock=unlimited:unlimited && source %s/setup.sh && xrt-smi examine\n' "$xrt_root"
printf 'This script did not run sudo, prlimit, or xrt-smi.\n'

exit "$xdna_status"
