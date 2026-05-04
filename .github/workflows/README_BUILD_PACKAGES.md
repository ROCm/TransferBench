# Building Relocatable TransferBench Packages with GitHub Actions

This document describes the GitHub Actions workflow for building relocatable
TransferBench packages using the ROCm SDK from
[TheRock](https://github.com/ROCm/TheRock).

The workflow (`.github/workflows/build-relocatable-packages.yml`) and the
`build_packages_local.sh` script at the repo root produce:

- **DEB** packages for Ubuntu/Debian
- **RPM** packages for AlmaLinux/Rocky/RHEL (built in `manylinux_2_28`)
- **TGZ** archives for any Linux distribution

All packages install to `/opt/rocm/extras-<MAJOR>` and use relocatable
`$ORIGIN`-relative `RPATH` so the install tree itself can be moved without
hard-coded library paths. These artifacts are **not** fully self-contained:
target systems must still provide the required ROCm/HSA runtime libraries
(declared as package dependencies: `hsa-rocr` and `numactl`).

This workflow is modeled on the
[ROCmValidationSuite packaging workflow](https://github.com/ROCm/ROCmValidationSuite/blob/master/.github/workflows/README_BUILD_PACKAGES.md).

## Workflow Triggers

| Trigger | Behavior |
|---------|----------|
| Push to `develop`, `mainline`, `release/**` | Build + upload to S3 (if configured) + regenerate apt/yum repo metadata |
| Pull request to `develop`, `mainline` | Build + upload to ref-specific S3 path (no repo metadata) |
| Schedule (daily 13:00 UTC) | Same as push, with auto-fetched latest ROCm |
| `workflow_dispatch` | Manual trigger with `rocm_version` and `gpu_family` inputs |

### Manual trigger inputs

- **`rocm_version`** (e.g. `7.11.0a20260121`). Empty = auto-fetch latest from TheRock.
- **`gpu_family`** — one of:
  - `gfx94X-dcgpu` (MI300A/MI300X) — **default**
  - `gfx950-dcgpu` (MI350X/MI355X)
  - `gfx110X-all` (RX 7900 XTX, 7800 XT, 7700S, Radeon 780M)
  - `gfx120X-all` (RX 9060/XT, 9070/XT)
  - `gfx1151` (Strix Halo iGPU)

## Build features enabled in CI

The workflow always builds with:

- `ENABLE_NIC_EXEC=OFF` — RDMA NIC executor disabled (would require libibverbs.so.1 at runtime; not bundled by TheRock SDK)
- `ENABLE_MPI_COMM=OFF` — MPI multi-node communicator disabled (would require OpenMPI at runtime; not bundled by TheRock SDK). Packages are built to run out of the box with only `numactl`/`libnuma1` from the OS.
- `DISABLE_DMABUF=OFF` — DMA-BUF support for GPU Direct RDMA
- `BUILD_RELOCATABLE_PACKAGE=ON` — RVS-style install prefix + package naming
- `GPU_TARGETS` — full data-center + consumer set (gfx906, 908, 90a, 942, 950, 1030, 1100/01/02, 1150/51, 1200/01)

## Local builds

The same script the workflow uses also works locally:

```bash
# Auto-fetch latest ROCm
sudo ./build_packages_local.sh

# Pin a specific version (use sudo -E to preserve env)
sudo -E ROCM_VERSION=7.11.0a20260121 GPU_FAMILY=gfx94X-dcgpu ./build_packages_local.sh

# Debug build
sudo -E BUILD_TYPE=Debug ./build_packages_local.sh
```

`sudo` is required because the script installs system packages
(`libnuma-dev`, `libibverbs-dev`, `libopenmpi-dev`, etc).

After the script completes, packages live under `build/`:

```
build/amdrocm7-transferbench_1.66.02-<release>_amd64.deb
build/amdrocm7-transferbench-1.66.02-<release>.x86_64.rpm
build/amdrocm7-transferbench-1.66.02-Linux.tar.gz
```

## Installing built packages

### Ubuntu / Debian

```bash
sudo dpkg -i build/amdrocm7-transferbench_*.deb
/opt/rocm/extras-7/bin/TransferBench
```

### Rocky / RHEL / AlmaLinux

```bash
sudo rpm -i --replacefiles --nodeps build/amdrocm7-transferbench-*.rpm
/opt/rocm/extras-7/bin/TransferBench
```

### Any Linux (TGZ — relocatable install tree, requires ROCm runtime on target)

End-user instructions (pre-install ROCm, runtime dependencies, extract,
`PATH` / `LD_LIBRARY_PATH`, troubleshooting) live in the project docs at
[docs/install/INSTALL_TGZ.rst](../../docs/install/INSTALL_TGZ.rst).

Quick smoke test from the repo root after a successful build:

```bash
sudo mkdir -p /opt/rocm/extras-7
sudo tar -xzf build/amdrocm7-transferbench-*.tar.gz -C /opt/rocm/extras-7 --strip-components=1
export PATH=/opt/rocm/extras-7/bin:$PATH
TransferBench --help
```

## S3 upload (OIDC)

S3 upload runs only when:
- The repository is `ROCm/TransferBench`, **and**
- The `AWS_S3_BUCKET` repository variable is set.

Upload uses **AWS OIDC** — no long-term keys are stored in the repo.

### S3 path layout

| Trigger | Path |
|---------|------|
| `release/*` push or dispatch | `release/transferbench/{deb,rpm,tar}/` |
| Schedule, push to `develop`/`mainline`, dispatch on non-release | `nightly/transferbench/{deb,rpm,tar}/` |
| Pull request (same repo) | `transferbench/<head_ref>/<run_number>/{ubuntu-22.04,manylinux_2_28}/` |

### Required repository setup

In **Settings → Secrets and variables → Actions**:

**Secrets tab:**
- `AWS_ROLE_ARN` — IAM role ARN with OIDC trust for this repo (e.g. `arn:aws:iam::123456789012:role/rocm-transferbench-s3-upload`)

**Variables tab:**
- `AWS_S3_BUCKET` — bucket name (e.g. `rocm-transferbench-packages`)
- `RUNNER_LABEL` (optional) — override Ubuntu runner label (default `ubuntu-22.04`)
- `RUNNER_LABEL_CONTAINER` (optional) — override container-job runner label (default `ubuntu-latest`)
- `RUNNER_LABEL_UTILITY` (optional) — override summary-job runner label (default `ubuntu-latest`)

### IAM role trust policy

The role in `AWS_ROLE_ARN` must trust GitHub's OIDC provider:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {
      "Federated": "arn:aws:iam::<ACCOUNT_ID>:oidc-provider/token.actions.githubusercontent.com"
    },
    "Action": "sts:AssumeRoleWithWebIdentity",
    "Condition": {
      "StringEquals": { "token.actions.githubusercontent.com:aud": "sts.amazonaws.com" },
      "StringLike":   { "token.actions.githubusercontent.com:sub": "repo:ROCm/TransferBench:*" }
    }
  }]
}
```

Permissions needed: `s3:PutObject`, `s3:GetObject`, `s3:ListBucket`, `s3:DeleteObject` on the bucket.

## Using the S3 paths as apt / yum repos

Push and scheduled builds also publish APT / YUM metadata so the S3 paths
work directly as native package repositories.

### apt (Ubuntu / Debian)

```bash
echo "deb [trusted=yes] https://<bucket>.s3.amazonaws.com/nightly/transferbench/deb/ ./" \
  | sudo tee /etc/apt/sources.list.d/transferbench-nightly.list
sudo apt update
sudo apt install amdrocm7-transferbench
```

### yum / dnf (Rocky / RHEL / AlmaLinux)

```bash
sudo tee /etc/yum.repos.d/transferbench-nightly.repo <<'EOF'
[transferbench-nightly]
name=TransferBench Nightly
baseurl=https://<bucket>.s3.amazonaws.com/nightly/transferbench/rpm/
enabled=1
gpgcheck=0
EOF
sudo dnf install amdrocm7-transferbench
```

> **Note:** `[trusted=yes]` / `gpgcheck=0` skip GPG verification. For
> production deployments, sign packages and metadata with a GPG key.

## Verifying RPATH

```bash
readelf -d /opt/rocm/extras-7/bin/TransferBench | grep -E 'RPATH|RUNPATH'
# Should contain $ORIGIN, $ORIGIN/../lib, /opt/rocm/extras-7/lib
```

## Troubleshooting

### S3 step fails with "Credentials could not be loaded"

- PR from a fork: OIDC is unavailable; the upload step is skipped.
- Same-repo: confirm `AWS_ROLE_ARN` secret is set and the role's trust
  policy allows `repo:ROCm/TransferBench:*`.

### Build fails: missing `libibverbs.h` / `mpi.h`

The packaged builds disable both `ENABLE_NIC_EXEC` and `ENABLE_MPI_COMM`, so these
headers are not required. If you've manually re-enabled either flag for a local
build, install the dev packages yourself:

```bash
# Ubuntu — for ENABLE_NIC_EXEC=ON
sudo apt install -y libibverbs-dev rdma-core
# Ubuntu — for ENABLE_MPI_COMM=ON
sudo apt install -y libopenmpi-dev openmpi-bin
# Rocky/RHEL
sudo dnf install -y rdma-core-devel openmpi-devel
```

### TheRock tarball download 404s

Check available builds at
<https://therock-nightly-tarball.s3.amazonaws.com/index.html>. Set
`ROCM_VERSION` explicitly to a known-good version.

## References

- [TheRock Releases](https://github.com/ROCm/TheRock/blob/main/RELEASES.md)
- [TheRock nightly tarballs](https://therock-nightly-tarball.s3.amazonaws.com/index.html)
- [ROCmValidationSuite packaging workflow](https://github.com/ROCm/ROCmValidationSuite/blob/master/.github/workflows/README_BUILD_PACKAGES.md) — reference implementation
- [TransferBench README](../../README.md)
