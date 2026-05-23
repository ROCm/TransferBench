#!/bin/sh
# Advisory check used as both the DEB postinst and the RPM %post scriptlet.
# TransferBench links against libhsa-runtime64.so.1; without it the binary
# fails at first run with a dynamic-linker error. The relocatable package
# declares no hard dep on hsa-rocr (or any ROCm component) because it is
# expected to install on TheRock-tarball systems where no ROCm package is
# tracked by apt/dpkg. Warn here so the user diagnoses a missing runtime
# at install time, not at TransferBench launch.
#
# Always exits 0 — this is advisory, never fatal.

set -e

found=0

if command -v ldconfig >/dev/null 2>&1; then
    if ldconfig -p 2>/dev/null | grep -q 'libhsa-runtime64\.so\.1'; then
        found=1
    fi
fi

if [ "$found" -eq 0 ]; then
    for d in /opt/rocm/lib /opt/rocm/lib64 /opt/rocm-*/lib /opt/rocm/extras-*/lib /opt/rocm/core-*/lib; do
        if [ -e "$d/libhsa-runtime64.so.1" ]; then
            found=1
            break
        fi
    done
fi

if [ "$found" -eq 0 ]; then
    cat >&2 <<'EOF'
====================================================================
TransferBench: WARNING

libhsa-runtime64.so.1 was not found on the dynamic loader path or
under any of /opt/rocm/lib, /opt/rocm-*/lib, or /opt/rocm/extras-*/lib.

TransferBench requires the ROCm HSA runtime at run time. Install a
ROCm 7.x stack (system packages or TheRock SDK) before invoking
TransferBench, or set LD_LIBRARY_PATH to a directory containing
libhsa-runtime64.so.1.

Without it, TransferBench will fail at startup with:
    error while loading shared libraries: libhsa-runtime64.so.1
====================================================================
EOF
fi

exit 0
