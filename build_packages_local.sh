#!/usr/bin/env bash
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# build_packages_local.sh — single source of truth for building relocatable
# TransferBench packages (DEB / RPM / TGZ) against TheRock ROCm SDK.
# Used by both local developers and the GitHub Actions workflow.
#
# Usage:
#   sudo ./build_packages_local.sh
#   sudo -E ROCM_VERSION=7.11.0a20260121 GPU_FAMILY=gfx94X-dcgpu ./build_packages_local.sh
#
# Requires root (installs system packages).

set -euo pipefail

# -------- pretty output --------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()   { echo -e "${GREEN}[ OK ]${NC}  $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()  { echo -e "${RED}[FAIL]${NC} $*" >&2; }

trap 'err "Build failed at line $LINENO"' ERR

# -------- inputs --------
ROCM_VERSION="${ROCM_VERSION:-}"                 # empty => auto-fetch latest
GPU_FAMILY="${GPU_FAMILY:-gfx94X-dcgpu}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
GITHUB_RUN_NUMBER="${GITHUB_RUN_NUMBER:-1}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${REPO_ROOT}/build"
SDK_DIR="${HOME}/rocm-sdk"
ROCM_PATH="${SDK_DIR}/install"

# Default GPU targets baked into every package, regardless of GPU_FAMILY tarball.
DEFAULT_GPU_TARGETS="gfx906;gfx908;gfx90a;gfx942;gfx950;gfx1030;gfx1100;gfx1101;gfx1102;gfx1150;gfx1151;gfx1200;gfx1201"
GPU_TARGETS="${GPU_TARGETS:-$DEFAULT_GPU_TARGETS}"

# -------- detect OS --------
if [[ -f /etc/os-release ]]; then
  # shellcheck disable=SC1091
  . /etc/os-release
  OS_ID="${ID:-unknown}"
  OS_LIKE="${ID_LIKE:-}"
else
  err "/etc/os-release not found; cannot detect distro"; exit 1
fi

case "${OS_ID}:${OS_LIKE}" in
  ubuntu:*|debian:*|*:*debian*)   DISTRO="ubuntu" ;;
  almalinux:*|rocky:*|rhel:*|centos:*|*:*rhel*|*:*fedora*) DISTRO="almalinux" ;;
  *)
    if command -v apt-get >/dev/null 2>&1; then DISTRO="ubuntu"
    elif command -v yum >/dev/null 2>&1 || command -v dnf >/dev/null 2>&1; then DISTRO="almalinux"
    else err "Unsupported distro: ${OS_ID}"; exit 1
    fi
    ;;
esac
log "Detected distro: ${DISTRO} (${OS_ID})"

# -------- install dependencies --------
log "Installing build dependencies..."
if [[ "${DISTRO}" == "ubuntu" ]]; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -y
  apt-get install -y --no-install-recommends \
    build-essential cmake git curl tar xz-utils ca-certificates pkg-config \
    python3 python3-pip \
    libnuma-dev libibverbs-dev rdma-core ibverbs-providers \
    libopenmpi-dev openmpi-bin \
    dpkg-dev rpm file apt-utils
  CMAKE_BIN="cmake"
  CMAKE_CXX_COMPILER_OVERRIDE=""
else
  # AlmaLinux / Rocky / RHEL / manylinux_2_28
  if command -v dnf >/dev/null 2>&1; then PKG="dnf"; else PKG="yum"; fi
  ${PKG} install -y epel-release || true
  # Enable PowerTools/CRB for createrepo_c, etc.
  ${PKG} config-manager --set-enabled powertools 2>/dev/null \
    || ${PKG} config-manager --set-enabled crb 2>/dev/null || true
  ${PKG} install -y \
    gcc gcc-c++ make cmake3 git curl tar xz ca-certificates pkgconfig \
    python3 python3-pip \
    numactl-devel rdma-core-devel libibverbs \
    openmpi-devel \
    rpm-build dpkg createrepo_c file
  CMAKE_BIN="cmake3"
  command -v cmake3 >/dev/null 2>&1 || CMAKE_BIN="cmake"
  CMAKE_CXX_COMPILER_OVERRIDE="${ROCM_PATH}/bin/hipcc"
  # OpenMPI on RHEL-likes ships under /usr/lib64/openmpi
  if [[ -d /usr/lib64/openmpi/bin ]]; then export PATH="/usr/lib64/openmpi/bin:${PATH}"; fi
  if [[ -d /usr/lib64/openmpi/lib ]]; then export LD_LIBRARY_PATH="/usr/lib64/openmpi/lib:${LD_LIBRARY_PATH:-}"; fi
fi
ok "Dependencies installed"

# -------- fetch ROCm SDK from TheRock --------
TAROBALL_BASE="https://therock-nightly-tarball.s3.amazonaws.com"
TAR_PREFIX="therock-dist-linux-${GPU_FAMILY}-"

if [[ -z "${ROCM_VERSION}" ]]; then
  log "ROCM_VERSION not set; auto-fetching latest for ${GPU_FAMILY}..."
  # No LATEST.txt is published; list the bucket and pick the highest version key.
  LIST_URL="${TAROBALL_BASE}/?list-type=2&max-keys=1000&prefix=${TAR_PREFIX}"
  LATEST_KEY="$(curl -fsSL "${LIST_URL}" 2>/dev/null \
    | tr '<' '\n' \
    | sed -n 's|^Key>||p' \
    | grep -E '\.tar\.gz$' \
    | sort -V \
    | tail -1 || true)"
  if [[ -n "${LATEST_KEY}" ]]; then
    ROCM_VERSION="${LATEST_KEY#${TAR_PREFIX}}"
    ROCM_VERSION="${ROCM_VERSION%.tar.gz}"
    ok "Latest ROCm version for ${GPU_FAMILY}: ${ROCM_VERSION}"
  else
    warn "Could not list ${LIST_URL}; falling back to pinned default"
    ROCM_VERSION="7.13.0a20260423"
  fi
fi

TARBALL_NAME="${TAR_PREFIX}${ROCM_VERSION}.tar.gz"
TARBALL_URL="${TAROBALL_BASE}/${TARBALL_NAME}"

mkdir -p "${SDK_DIR}"
if [[ ! -d "${ROCM_PATH}" ]] || [[ ! -f "${SDK_DIR}/.installed-${ROCM_VERSION}-${GPU_FAMILY}" ]]; then
  log "Downloading ${TARBALL_URL}..."
  curl -fSL "${TARBALL_URL}" -o "${SDK_DIR}/${TARBALL_NAME}"
  log "Extracting to ${SDK_DIR}..."
  rm -rf "${ROCM_PATH}"
  mkdir -p "${ROCM_PATH}"
  tar -xzf "${SDK_DIR}/${TARBALL_NAME}" -C "${ROCM_PATH}" --strip-components=1 \
    || tar -xzf "${SDK_DIR}/${TARBALL_NAME}" -C "${ROCM_PATH}"
  rm -f "${SDK_DIR}/${TARBALL_NAME}"
  touch "${SDK_DIR}/.installed-${ROCM_VERSION}-${GPU_FAMILY}"
  ok "ROCm SDK installed at ${ROCM_PATH}"
else
  log "Reusing cached ROCm SDK at ${ROCM_PATH}"
fi

export ROCM_PATH
export PATH="${ROCM_PATH}/bin:${PATH}"
export LD_LIBRARY_PATH="${ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}"
export CMAKE_PREFIX_PATH="${ROCM_PATH}:${CMAKE_PREFIX_PATH:-}"

# Locate HIP device libraries (amdgcn bitcode)
for candidate in \
  "${ROCM_PATH}/amdgcn/bitcode" \
  "${ROCM_PATH}/lib/llvm/amdgcn/bitcode" \
  "${ROCM_PATH}/lib/clang/amdgcn/bitcode"; do
  if [[ -d "${candidate}" ]]; then export HIP_DEVICE_LIB_PATH="${candidate}"; break; fi
done
if [[ -n "${HIP_DEVICE_LIB_PATH:-}" ]]; then
  ok "HIP_DEVICE_LIB_PATH=${HIP_DEVICE_LIB_PATH}"
else
  warn "amdgcn bitcode directory not found under ${ROCM_PATH}; build may fail"
fi

# -------- compute version helpers --------
# ROCM_MAJOR / MINOR / patch helpers (e.g. 7.11.0a20260121 -> major=7 minor=11)
ROCM_MAJOR="$(echo "${ROCM_VERSION}" | sed -E 's/^([0-9]+)\..*/\1/')"
ROCM_MINOR="$(echo "${ROCM_VERSION}" | sed -E 's/^[0-9]+\.([0-9]+).*/\1/')"
printf -v ROCM_LIBPATCH_VERSION '%02d%02d' "${ROCM_MAJOR}" "${ROCM_MINOR}"
export ROCM_MAJOR ROCM_MINOR ROCM_LIBPATCH_VERSION
log "ROCm major=${ROCM_MAJOR} minor=${ROCM_MINOR} libpatch=${ROCM_LIBPATCH_VERSION}"

# Package release string: branch.commit for dev, run_number for release branches
GIT_BRANCH="${GITHUB_REF_NAME:-$(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)}"
GIT_COMMIT="$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
if [[ "${GIT_BRANCH}" == rel* ]] || [[ "${GIT_BRANCH}" == release/* ]]; then
  PKG_RELEASE="${GITHUB_RUN_NUMBER}"
else
  PKG_RELEASE="${GIT_BRANCH//\//.}.${GIT_COMMIT}"
fi
export CPACK_DEBIAN_PACKAGE_RELEASE="${CPACK_DEBIAN_PACKAGE_RELEASE:-$PKG_RELEASE}"
export CPACK_RPM_PACKAGE_RELEASE="${CPACK_RPM_PACKAGE_RELEASE:-$PKG_RELEASE}"
log "Package release tag: ${PKG_RELEASE}"

# -------- configure --------
INSTALL_PREFIX="/opt/rocm/extras-${ROCM_MAJOR}"
RPATH_LIST="\$ORIGIN:\$ORIGIN/../lib:${INSTALL_PREFIX}/lib:${ROCM_PATH}/lib"

log "Configuring CMake..."
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"

CMAKE_ARGS=(
  -B "${BUILD_DIR}"
  -S "${REPO_ROOT}"
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
  -DROCM_PATH="${ROCM_PATH}"
  -DROCM_MAJOR_VERSION="${ROCM_MAJOR}"
  -DHIP_PLATFORM=amd
  -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}"
  -DCPACK_PACKAGING_INSTALL_PREFIX="${INSTALL_PREFIX}"
  -DCMAKE_SKIP_RPATH=FALSE
  -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=FALSE
  -DCMAKE_INSTALL_RPATH="${RPATH_LIST}"
  -DCMAKE_VERBOSE_MAKEFILE=ON
  -DBUILD_RELOCATABLE_PACKAGE=ON
  -DBUILD_LOCAL_GPU_TARGET_ONLY=OFF
  -DENABLE_NIC_EXEC=ON
  -DENABLE_MPI_COMM=ON
  -DDISABLE_DMABUF=OFF
  -DGPU_TARGETS="${GPU_TARGETS}"
)
if [[ -n "${CMAKE_CXX_COMPILER_OVERRIDE}" ]]; then
  CMAKE_ARGS+=(-DCMAKE_CXX_COMPILER="${CMAKE_CXX_COMPILER_OVERRIDE}")
fi

"${CMAKE_BIN}" "${CMAKE_ARGS[@]}"
ok "CMake configured"

# -------- build --------
log "Building TransferBench (-j$(nproc))..."
"${CMAKE_BIN}" --build "${BUILD_DIR}" -- -j"$(nproc)"
ok "Build complete"

# -------- package --------
log "Packaging (DEB / RPM / TGZ via CPack)..."
pushd "${BUILD_DIR}" >/dev/null
if [[ "${DISTRO}" == "ubuntu" ]]; then
  cpack -G DEB
  cpack -G TGZ
else
  cpack -G RPM
  cpack -G TGZ
fi
popd >/dev/null

ok "Packages written under ${BUILD_DIR}:"
ls -lh "${BUILD_DIR}"/amdrocm*-transferbench* 2>/dev/null || ls -lh "${BUILD_DIR}"/*.deb "${BUILD_DIR}"/*.rpm "${BUILD_DIR}"/*.tar.gz 2>/dev/null || true
