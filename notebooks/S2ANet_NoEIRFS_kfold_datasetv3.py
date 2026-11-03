# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # S2A-Net (Han et al., TGRS 2021) -- No E-IRFS (ablation) + embedded HPO + K-fold final training, datasetv3
#
# https://github.com/csuhan/s2anet / https://arxiv.org/abs/2008.09397
#
# **Ablation twin of `notebooks/S2ANet_E-IRFS_kfold_datasetv3.py`** -- isolates
# E-IRFS's effect on this arm's results, same convention as the YOLO11m/
# YOLO26m `*_NoEIRFS_kfold_datasetv3.py` twins. `USE_EIRFS = False` below is
# threaded through to `src/s2anet_worker.py`, which then skips
# `attach_eirfs()` entirely and trains on mmdet's own stock (unweighted)
# sampling instead.
#
# **Deliberate deviation from every other arm's NoEIRFS/E-IRFS pair, by
# request:** this notebook runs its **own** embedded Optuna search
# (train-only folds) rather than loading `best_params.json` from the E-IRFS
# notebook's search. Every other NoEIRFS twin in this project (YOLO11m,
# YOLO26m) deliberately loads the E-IRFS run's own hyperparameters instead,
# specifically so E-IRFS presence/absence is the *only* varying factor
# between the two notebooks' results. Running a separate search here means
# this arm's E-IRFS-vs-NoEIRFS comparison is **no longer isolated to just
# E-IRFS** -- any accuracy difference could also come from the two searches
# landing on different hyperparameters. Accepted trade-off; flag this when
# reporting S2A-Net's ablation numbers. Everything else (fold assignment/
# seed, epochs, architecture, config fields) is kept byte-identical to the
# E-IRFS notebook.
#
# This notebook only trains/tests -- it does **not** include a "re-test an
# already-archived run without retraining" section, unlike the other arms'
# K-fold notebooks. That flow lives in its own dedicated
# `notebooks/S2ANet_E-IRFS_retest_datasetv3.py` instead, so this file can run
# start-to-finish non-interactively (e.g. via a Colab CLI runner) without a
# manual "skip this section" decision partway through.
#
# ## Environment warning -- read before running
#
# S2A-Net's custom CUDA ops (box-IoU-rotated, rotated NMS, the alignment
# convolution) are pinned to **PyTorch 1.3 + CUDA 10.0** (the repo's own
# `docs/INSTALL.md`: "PyTorch 1.3 or higher... CUDA 9.0 or higher", tested
# against 1.3.1/CUDA 10.0-10.1) and `mmcv==0.2.14` -- a pre-`mmcv-full` split
# release, far older than Strip R-CNN's `mmcv-full==1.7.2`/`mmdet==2.28.2`.
# Like Strip R-CNN, this notebook's own Colab kernel cannot import that stack
# directly (no `torch==1.3` wheel exists for Python 3.11+), so it installs a
# **separate Miniconda `s2anet` env pinned to Python 3.7** and routes all
# actual training/inference through it via subprocess -- see
# `src/s2anet_worker.py` and the "file setup" section below.
#
# **This is a harder constraint than Strip R-CNN's, not just an older one,
# and this is the single least-verifiable step in this whole notebook** --
# unlike everything else here, checked against S2A-Net's/mmdet's/mmcv's
# actual source, this specific build step could not be executed end-to-end
# before you run it; expect it may take one round of live debugging.
#
# CUDA 10.0's `nvcc` has no concept of the Ampere (`sm_80`, e.g. A100) or Ada
# (`sm_89`, e.g. L4) architectures at all -- they did not exist yet in 2019 --
# so building against a GPU newer than Turing (`sm_75`, e.g. T4) cannot work
# regardless of toolchain -- and neither can running there, since cuDNN 7.6's
# kernels are closed precompiled binaries with no PTX to JIT from, so the first
# convolution fails with `CUDNN_STATUS_EXECUTION_FAILED`. **Turing or older is
# the requirement; a T4 is the easy choice.** A V100 (`sm_70`), P100 (`sm_60`)
# or K80 (`sm_37`) is equally fine -- the ops are compiled for whatever card is
# attached (see the ops-build cell). What cannot work is anything NEWER: L4
# (`sm_89`), A100/A10G (`sm_80`/`sm_86`), RTX A6000 or RTX 6000 Ada (`sm_86`/
# `sm_89`), or any Blackwell card such as the RTX PRO 6000 (`sm_120`). Note the
# naming trap: a *Quadro* RTX 6000 is Turing and works; an *RTX PRO* 6000 is
# Blackwell and does not. **If your runtime hands you one of those, request a
# T4 instead** (Runtime -> Change runtime type, if your plan offers a choice)
# before running this notebook.
#
# Separately, even on a T4: `python setup.py develop` compiles S2A-Net's
# custom CUDA ops via `torch.utils.cpp_extension.BuildExtension`, which needs
# an actual `nvcc` + host C++ compiler on `PATH` -- installing only conda's
# `cudatoolkit=10.0` (the runtime libraries) does NOT provide one, so without
# also installing a matching `cudatoolkit-dev`/`nvcc` and an old-enough host
# `g++` (CUDA 10.0's documented supported range tops out around GCC 7),
# the build would silently fall back to whatever `nvcc`/`g++` Colab's system
# image ships by default -- almost certainly much newer than what CUDA
# 10.0-era source and torch 1.3's own C++ headers were written against,
# which is exactly the kind of mismatch that broke Oriented-DETR's
# `ms_deform_attn_cuda.cu` earlier in this project (see CLAUDE.md), except
# with a much bigger version gap here. The cells below install an
# old-enough host `g++` (7.x, from conda-forge) plus `nvcc` itself.
#
# **`nvcc` does NOT come from conda here, deliberately.** conda-forge's
# `cudatoolkit-dev=10.0` package is the obvious way to get CUDA 10.0's
# `nvcc`, and it is what this notebook originally used -- but it is broken
# against every current conda release, confirmed live: its post-link script
# does `from conda.exports import download, hashsum_file`, and
# `hashsum_file` no longer exists in `conda.exports` (verified against
# conda's own current source). The package has not been updated for that
# removal, so the install always fails with an `ImportError` and conda
# rolls the whole transaction back -- which also silently takes
# `pytorch`/`torchvision` down with it if they are in the same transaction.
# No install flag, channel pin, or ordering change fixes this. So the cell
# below downloads NVIDIA's own CUDA 10.0 runfile and runs its toolkit-only
# silent install instead, which is what `cudatoolkit-dev`'s post-link
# script was itself trying to do anyway, minus the broken conda API call.

# %% [markdown]
# ## file setup

# %%
# The cheapest possible gate, and deliberately the FIRST thing in file setup:
# everything below it -- a conda env, a 2 GB CUDA 10.0 runfile, an mmcv source
# build, the s2anet ops build -- is wasted effort on the wrong GPU, and the
# failure would otherwise not surface until tens of minutes into training, as
# CUDNN_STATUS_EXECUTION_FAILED in the first convolution. Exactly that happened
# on an L4: ~38 minutes of PTX JIT per fold, three folds, then death. See the
# env-check cell below, and the header, for why the ceiling is Turing.
#
# nvidia-smi rather than torch, because no torch exists in this kernel at all
# (that is the whole point of the conda env below), so this runs before any of
# the setup rather than after it.
import subprocess as _sp


def _smi(fields):
    try:
        return _sp.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"],
                       capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:      # no GPU runtime attached at all
        return ""


# `compute_cap` needs a reasonably recent driver. When it is available the
# capability is compared numerically -- anything Turing or older is fine, and
# the ops build below targets whatever is attached. When it is not, this does
# not guess from the marketing name (a *Quadro* RTX 6000 is Turing and works, an
# *RTX PRO* 6000 is Blackwell and does not): it only insists a GPU exists, and
# leaves the authoritative check to the env-check cell, which reads the
# capability through torch itself.
_raw = _smi("name,compute_cap")
if "," in _raw:
    _name, _cap = [s.strip() for s in _raw.rsplit(",", 1)]
    print(f"GPU: {_name} (compute capability {_cap})")
    assert tuple(int(v) for v in _cap.split(".")) <= (7, 5), (
        f"{_name} is compute capability {_cap}; this arm needs Turing or older "
        "(<= 7.5, e.g. a T4). Runtime -> Change runtime type -> T4 GPU, then "
        "re-run this cell. CUDA 10.0 and cuDNN 7.6 ship no kernels for anything "
        "past Turing, and cuDNN's are closed binaries it cannot JIT from PTX, so "
        "a newer card spends tens of minutes JIT-ing torch's PTX and then fails "
        "in the first convolution with CUDNN_STATUS_EXECUTION_FAILED."
    )
else:
    print("GPU:", _raw or "NONE DETECTED",
          "\n  (could not read compute capability from nvidia-smi here -- the "
          "env-check cell below verifies it through torch, before the ops build)")
    assert _raw, ("No GPU detected. Runtime -> Change runtime type -> T4 GPU. "
                  "This arm needs Turing or older; see the header.")

# %%
# This notebook's own Colab kernel installs only what it needs to run
# framework-free code (Optuna, the shared src/obb_eval.py evaluator, etc.).
# The s2anet stack itself (torch==1.3, mmcv==0.2.14, Python 3.7) is NOT
# installed here -- see the next cells for why and how.
# !pip3 install -q optuna pandas numpy pyyaml shapely

# %%
# Miniconda + a Python 3.7 `s2anet` env, same architecture as the Strip
# R-CNN notebooks' "file setup" section (see CLAUDE.md's "Strip R-CNN:
# Miniconda + subprocess-worker architecture" entry) -- installed once per
# runtime; re-running this cell in an already-set-up runtime is mostly a
# fast no-op via conda/pip's own package caching.
# !wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
# !bash /tmp/miniconda.sh -b -p /content/miniconda -u
# Anaconda now requires explicit Terms-of-Service acceptance for the
# default channels before `conda create` will run non-interactively --
# without this, `conda create` fails with CondaToSNonInteractiveError, the
# s2anet env is never actually created, and every cell below it then fails
# with "EnvironmentLocationNotFound: Not a conda environment", which only
# surfaces much later as a bare FileNotFoundError when training tries to
# launch the conda-env subprocess. Idempotent -- a no-op if already
# accepted. Same fix as the Strip R-CNN notebooks' own "file setup" section.
# !/content/miniconda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
# !/content/miniconda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
# !/content/miniconda/bin/conda create -y -n s2anet python=3.7
# pytorch, torchvision, the CUDA 10.0 RUNTIME libraries (libcudart etc., what
# torch links against at import/run time), the old host compiler, and swig --
# all into the s2anet env. `cudatoolkit-dev` (i.e. `nvcc`) is deliberately
# NOT here: it is broken against current conda, see the header and the next
# cell. Nothing is installed into conda's BASE env on purpose -- an earlier
# attempt to install cudatoolkit-dev there worked around one failure but
# silently upgraded conda itself as a dependency, which is exactly the kind
# of base-env churn worth avoiding.
#
# **The torch stack comes from DEFAULTS ONLY -- deliberately no `-c pytorch`
# and no `-c conda-forge` on that line.** Confirmed live why this matters:
# with `-c pytorch -c conda-forge` and an unpinned `torchvision`, the solver
# picked conda-forge's `torchvision-0.10.1-*_cpu` (0.10 belongs with torch
# 1.9, not 1.3) and that dragged in `pytorch-cpu 1.1.0` as a SECOND torch
# writing into the same site-packages/torch. conda reports no conflict --
# `pytorch` and `pytorch-cpu` are separate package names -- but the result is
# an env where `torch.cuda.is_available()` is False, which is what torch's
# own "No CUDA runtime is found" line in the build log actually means (that
# message is printed precisely when cuda is unavailable, not merely when
# CUDA_HOME had to be guessed). Pinning pytorch to its CUDA build stops that.
#
# **torchvision is NOT installed by conda either, for a related reason.**
# Pinning pytorch alone does not constrain it: confirmed live, the solver
# then picked defaults' `torchvision-0.13.1-cpu_py37*`, which needs torch
# >=1.12 and blows up on `import torchvision` with "module 'torch.jit' has no
# attribute '_script_if_tracing'". conda simply has no py37 torchvision build
# that pairs with torch 1.3.1 (checked anaconda.org: 0.4.2 ships linux-64
# py27 builds only). PyPI does have the exact pair, so it comes from pip
# below with `--no-deps` -- without `--no-deps` pip would pull its own
# `torch==1.3.1` wheel, which is a CUDA 10.1 build and would clobber conda's
# cuda100 one.
# !/content/miniconda/bin/conda install -y -n s2anet "pytorch=1.3.1=cuda100py37*" cudatoolkit=10.0
# Build toolchain and headers, from conda-forge.
#
# **`sysroot_linux-64=2.17` is pinned and that pin is essential.** Left
# unpinned, conda resolves sysroot 2.39 next to gcc 7.5, and the two cannot
# work together: glibc 2.39's shared libraries use RELR relocations, which
# this toolchain's binutils 2.36 linker cannot parse, so every link step dies
# with "unknown type [0x13] section `.relr.dyn'" -> "skipping incompatible
# .../libm.so.6" -> "cannot find /lib64/libm.so.6" (confirmed live, on both
# the s2anet ops build and DOTA_devkit's polyiou). 2.17 is the CentOS-7-era
# sysroot that conda-forge's gcc 7.x toolchains were actually built against.
#
# `libxcrypt` is here for a non-obvious reason: Python 3.7's own `Python.h`
# does an unconditional `#include <crypt.h>`, but glibc moved `crypt.h` out
# into libxcrypt years ago, and nothing else in this env pulls it in -- so
# compiling ANY C/C++ Python extension below (s2anet's CUDA ops,
# DOTA_devkit's polyiou) dies with "fatal error: crypt.h: No such file or
# directory". Confirmed live, and confirmed it was a genuinely missing header
# rather than a missing include path: the failure persisted after conda
# activation was already supplying `-isystem $PREFIX/include`.
# !/content/miniconda/bin/conda install -y -n s2anet -c conda-forge gxx_linux-64=7 gcc_linux-64=7 sysroot_linux-64=2.17 swig libxcrypt
# **UNPREFIXED `gcc`/`g++` symlinks, and they are what make nvcc usable at
# all.** conda's compiler packages install only TRIPLET-PREFIXED binaries
# (`x86_64-conda-linux-gnu-gcc`) and hand those to builds through the
# activation-set `CC`/`CXX` -- deliberately, so an activated env never shadows
# the system `gcc`. But nvcc takes its host compiler from neither: it has no
# `CC`/`CXX`/`CUDAHOSTCXX` support at all here (`CUDAHOSTCXX` is a CMake
# variable, and torch 1.3.1's setuptools-based `cpp_extension.py` never reads
# it nor emits `-ccbin` -- confirmed by reading that file's actual source), so
# it falls back to its own built-in default: whatever plain `gcc`/`g++` it
# finds on PATH. On Colab that is the modern system gcc, and CUDA 10.0's
# `crt/host_config.h` refuses it outright with "#error -- unsupported GNU
# version! gcc versions later than 7 are not supported!" -- confirmed live,
# and confirmed to be the SYSTEM gcc rather than conda's by its diagnostic
# format (gcc 9+ prints line-numbered `129 | #error`; every conda-gcc-7
# message in the same build log uses the old unnumbered format).
#
# `conda run` puts `$PREFIX/bin` first on PATH, so a symlink there is what
# nvcc resolves. Guarded with `test -x` rather than `ln -sf` so that if a
# future conda ever does ship a real unprefixed `gcc`, this cannot clobber it.
# !test -x /content/miniconda/envs/s2anet/bin/gcc || ln -s /content/miniconda/envs/s2anet/bin/x86_64-conda-linux-gnu-gcc /content/miniconda/envs/s2anet/bin/gcc
# !test -x /content/miniconda/envs/s2anet/bin/g++ || ln -s /content/miniconda/envs/s2anet/bin/x86_64-conda-linux-gnu-g++ /content/miniconda/envs/s2anet/bin/g++
# torchvision 0.4.2 is the exact pair for torch 1.3.1 (its own metadata
# requires `torch==1.3.1`). `--no-deps` for the reason in the conda comment
# above. Its compiled `_C` extension is built against PyPI's CUDA 10.1 torch
# and may therefore be unloadable here -- that is fine and expected: nothing
# in this pipeline calls torchvision's own ops, and 0.4.2 loads `_C` lazily
# (inside functions, not at import), so `import torchvision` still succeeds.
#
# What torchvision is actually needed FOR here is only pure Python: mmcv
# 0.2.14 imports it at module level in mmcv/runner/checkpoint.py, and
# `build_config()` below sets `pretrained='torchvision://resnet50'`, which
# mmcv resolves via its own `get_torchvision_models()` -- that walks
# `torchvision.models` submodules collecting each one's `model_urls` dict
# (confirmed by reading mmcv 0.2.14's actual checkpoint.py). Do NOT "just
# upgrade" torchvision to fix some unrelated warning: those `model_urls`
# dicts were deprecated in 0.13 and REMOVED in 0.15+ in favour of the
# `Weights` enum API, so a modern torchvision makes that lookup raise
# KeyError and every fold dies at model init.
# !/content/miniconda/bin/conda run -n s2anet pip install -q --no-deps torchvision==0.4.2
# Cython and numpy FIRST, in their own pip call, and mmcv with
# `--no-build-isolation` after them: mmcv 0.2.14's setup.py does
# `from Cython.Distutils import build_ext` and `import numpy` at MODULE level
# (confirmed by reading that file's actual source), so it cannot build unless
# both are importable at build time -- and pip's default build isolation
# hides them even once they are installed. mmcv tries to self-heal this via
# the long-deprecated `fetch_build_eggs`, which is not something to rely on.
# !/content/miniconda/bin/conda run -n s2anet pip install -q cython numpy
# !/content/miniconda/bin/conda run -n s2anet pip install -q --no-build-isolation "mmcv==0.2.14"
# !/content/miniconda/bin/conda run -n s2anet pip install -q albumentations imagecorruptions matplotlib terminaltables six pycocotools "pillow==6.2.2" shapely

# %%
# `nvcc` for CUDA 10.0, straight from NVIDIA's own runfile installer rather
# than conda-forge's broken `cudatoolkit-dev=10.0` (see the header's
# "nvcc does NOT come from conda here" note for the confirmed reason).
#
# **Use the developer.nvidia.com URL, and do not "simplify" it to a direct
# developer.download.nvidia.com path.** For CUDA 10.0 the download is
# token-signed: developer.nvidia.com 302-redirects to a
# developer.download.nvidia.com URL carrying an `__token__` query parameter,
# and the unsigned path 404s. An earlier version of this cell used the
# unsigned path WITH `wget -q`, so the 404 was completely silent, nothing was
# installed, and the only symptom was torch later reporting
# "No CUDA runtime is found, using CUDA_HOME='/content/cuda-10.0'" -- a line
# that looks like routine noise. Note also the real filename has NO `.run`
# suffix. `wget` follows the redirect on its own; `-nv` keeps the ~2GB
# progress output short without hiding errors the way `-q` did.
#
# Flags: `--toolkit` installs ONLY the CUDA toolkit, never the bundled 410.48
# GPU driver -- Colab already has its own, much newer driver, and letting a
# 2018 driver anywhere near it would break the runtime. `--toolkitpath` keeps
# it in /content (Colab's own local disk, wiped per runtime like everything
# else here) rather than writing into /usr/local. `--override` skips the
# installer's own host-compiler version check, which would otherwise refuse
# to run at all against Colab's modern system gcc -- harmless here because
# the actual compiles below never use that system compiler: they run under
# conda's own gcc/g++ 7.x -- `CC`/`CXX` from env activation for the plain
# C/C++ steps, and the unprefixed `gcc`/`g++` symlinks the install cell above
# creates for nvcc's own host-compiler lookup.
#
# This is a ~2GB download plus extraction, so it is the slowest cell in the
# whole "file setup" section -- guarded so re-running it in a runtime that
# already has nvcc is an instant no-op instead of another 2GB.
# !test -x /content/cuda-10.0/bin/nvcc || (wget -nv https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_410.48_linux -O /tmp/cuda10.run && sh /tmp/cuda10.run --silent --toolkit --toolkitpath=/content/cuda-10.0 --override)
# Fail loudly here rather than letting a missing nvcc surface much later as a
# confusing compile error inside setup.py -- or, as happened before, as
# nothing at all until training tried to use ops that were never built.
# !/content/cuda-10.0/bin/nvcc --version

# %% [markdown]
# ### Environment check -- run this before the builds below
#
# One cheap cell that verifies everything the expensive steps below depend on.
# Every assertion here corresponds to a failure this notebook actually hit on
# a live Colab run, each of which originally surfaced *far* from its cause:
# a missing conda env as a bare `FileNotFoundError` three folds into HPO; a
# missing `crypt.h` as a compiler error twenty lines into a build log; a
# CPU-only torch as nothing but a "No CUDA runtime is found" warning that is
# easy to read as boilerplate. Catching them here costs seconds; catching
# them where they used to surface costs a full runtime setup.

# %%
# !test -f /content/miniconda/envs/s2anet/include/crypt.h && echo "crypt.h: OK"
# !test -x /content/cuda-10.0/bin/nvcc && echo "nvcc: OK"
# swig is a hard requirement, not an optional nicety: this repo ships no
# pre-generated polyiou_wrap.cxx, so the DOTA_devkit build below cannot run
# without it (see that cell's comment).
# !test -x /content/miniconda/envs/s2anet/bin/swig && echo "swig: OK"
# !/content/miniconda/bin/conda list -n s2anet | grep -iE "^(pytorch|torchvision|cudatoolkit|sysroot|gcc_linux|gxx_linux) " || true
# Does the toolchain actually LINK? This is the cheap version of the failure
# that otherwise only appears at the end of a multi-minute build: an
# unpinned sysroot gives a glibc whose RELR relocations binutils 2.36 cannot
# read, and every link dies with "cannot find /lib64/libm.so.6".
# !cd /tmp && printf '#include <math.h>\nint main(void){return (int)sqrt(4.0);}\n' > _linkprobe.c && /content/miniconda/bin/conda run -n s2anet --no-capture-output x86_64-conda-linux-gnu-gcc _linkprobe.c -lm -o _linkprobe && echo "toolchain link: OK"
# Which `gcc` does a plain, unprefixed call resolve to inside the activated
# env -- i.e. the one nvcc itself will pick up? Must report 7.x.
# !/content/miniconda/bin/conda run -n s2anet --no-capture-output gcc --version | head -1
# Does nvcc actually ACCEPT that host compiler? `cuda_runtime.h` pulls in
# `crt/host_config.h`, whose `#if __GNUC__ > 7` guard is the exact thing that
# killed a full ops build with "unsupported GNU version! gcc versions later
# than 7 are not supported!" -- for ~2 seconds here instead of several minutes
# into `setup.py develop`. Same `-gencode` as the real build, so this also
# proves CUDA 10.0's nvcc accepts the arch the real build will target -- derived
# from the attached GPU here exactly as it is there, so the two cannot drift.
# !cd /tmp && ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)"; test -n "$ARCH" || ARCH=7.5; SM="$(echo $ARCH | tr -d .)"; printf '#include <cuda_runtime.h>\nint main(void){return 0;}\n' > _nvccprobe.cu && /content/miniconda/bin/conda run -n s2anet --no-capture-output /content/cuda-10.0/bin/nvcc -gencode=arch=compute_$SM,code=sm_$SM -c _nvccprobe.cu -o _nvccprobe.o && echo "nvcc accepts sm_$SM: OK"
# torch must see the GPU, and torchvision must both import AND still expose
# the `model_urls` dicts that mmcv 0.2.14 uses to resolve
# `pretrained='torchvision://resnet50'`.
#
# The GPU assert goes FIRST, before any cosmetic printing. That ordering is
# not stylistic: an earlier version of this check printed
# `torchvision.__version__` first, which torchvision 0.4.2 does not define,
# so the check died with a misleading AttributeError and never reached the
# assert -- and the real problem (no GPU attached to the runtime) then
# surfaced minutes later, deep inside the build, as torch's
# "Found no NVIDIA driver on your system".
#
# The compute-capability assert is the second gate, and it must come before
# the conv probe at the end of the same line -- which is why all of this is
# ONE python -c and not several `!` lines: each `!` line is its own shell, so
# a failing one would not stop the next from running. Reading device
# properties launches no kernel and returns in milliseconds, whereas the first
# real convolution on an unsupported card costs tens of minutes of PTX JIT
# before cuDNN fails. That is not hypothetical: on an L4 this arm spent ~38
# minutes per fold JIT-ing and then died in the first conv2d with
# CUDNN_STATUS_EXECUTION_FAILED, three folds in a row, before this assert
# existed.
#
# The conv probe at the end runs the exact operation that failed there -- a
# real cuDNN convolution on the GPU -- so an arch/cuDNN mismatch surfaces here
# in seconds instead of in the first forward pass of the first HPO fold.
# !/content/miniconda/bin/conda run -n s2anet --no-capture-output python -c "import torch, torchvision; assert torch.cuda.is_available(), 'torch cannot see a GPU. Attach a GPU runtime (Runtime -> Change runtime type -> T4 GPU). If a GPU IS attached, check that ONLY ONE torch is installed -- a stray pytorch-cpu alongside pytorch broke this before.'; cap = torch.cuda.get_device_capability(); assert cap <= (7, 5), 'GPU is sm_%d%d (%s), but this arm needs Turing or older (capability <= 7.5, e.g. a T4): Runtime -> Change runtime type -> T4 GPU. CUDA 10.0 and cuDNN 7.6 shipped no kernels past Turing at all, and cuDNN cannot JIT from PTX, so on a newer card (L4 sm_89, A100 sm_80, RTX PRO 6000 sm_120) the first conv2d dies with CUDNN_STATUS_EXECUTION_FAILED after tens of minutes spent JIT-ing torch own PTX. Older is fine -- the ops build targets whatever card is attached.' % (cap[0], cap[1], torch.cuda.get_device_name(0)); from torchvision.models.resnet import model_urls; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| device', torch.cuda.get_device_name(0), '| capability', cap); print('resnet50 weights URL:', model_urls['resnet50']); y = torch.nn.Conv2d(3, 8, 3).cuda()(torch.randn(2, 3, 32, 32).cuda()); torch.cuda.synchronize(); print('cudnn conv:', tuple(y.shape), '| cudnn', torch.backends.cudnn.version())"
# mmcv 0.2.14 has no wheel and is built from source here under
# `--no-build-isolation` (see the install cell) -- a fragile enough step that
# it is worth proving it actually imports before spending minutes on the ops
# build below, rather than discovering it from the worker three folds deep.
# !/content/miniconda/bin/conda run -n s2anet --no-capture-output python -c "import mmcv; print('mmcv', mmcv.__version__)"

# %%
# Clone S2A-Net (this repo is itself a fork of mmdetection v1.x -- its own
# top-level package IS `mmdet`, which is why src/s2anet_worker.py's `import
# mmdet` resolves to THIS clone rather than a separately pip-installed mmdet
# from PyPI; nothing above installs a PyPI `mmdet`, on purpose) and build its
# own custom CUDA ops (box_iou_rotated, nms_rotated, the alignment
# convolution, active rotating filters, etc.) against the s2anet env's own
# torch 1.3/cudatoolkit 10.0 -- `setup.py develop` is this repo's own
# documented install command (docs/INSTALL.md).
#
# **The build runs through `conda run -n s2anet`, not the env's python binary
# directly, and that matters.** Invoking `.../envs/s2anet/bin/python setup.py
# develop` skips conda's ACTIVATION scripts -- and conda's compiler packages
# (gcc_linux-64/gxx_linux-64) ship their whole build environment in those
# scripts: CC, CXX, and critically CFLAGS/CPPFLAGS/LDFLAGS carrying
# `-I$PREFIX/include` and `-L$PREFIX/lib`. Without activation the compile line
# only gets `-I$PREFIX/include/python3.7m`, so Python 3.7's own `Python.h`
# (which does `#include <crypt.h>`, a header that lives at
# `$PREFIX/include/crypt.h` since glibc moved crypt out to libxcrypt) fails
# with "fatal error: crypt.h: No such file or directory" -- confirmed live.
# `conda run` activates the env properly and hands the compiler all of it.
# This is the same missing-activation root cause as the full-path `swig`
# below, fixed structurally here rather than by hand-patching flag by flag.
#
# Only CUDA_HOME stays explicit on top of activation: it points at the
# NVIDIA-runfile CUDA 10.0 install from the cell above, which does not come
# from conda, so activation knows nothing about it.
#
# **There is deliberately no `CUDAHOSTCXX` here, even though nvcc does need
# to be pointed at conda's old g++ 7.x rather than Colab's modern system
# one.** An earlier version of this cell set `CUDAHOSTCXX` and it had zero
# effect: that variable is a CMake one, and torch 1.3.1's setuptools-based
# `cpp_extension.py` neither reads it nor passes `-ccbin` (confirmed by
# reading that file). nvcc therefore always takes its host compiler from
# plain `gcc`/`g++` on PATH, which is why the install cell above creates
# unprefixed symlinks in the env's own `bin/` -- see that cell's comment. The
# version pairing genuinely matters beyond nvcc's own `#error` guard: torch
# 1.3's C++ headers were written against a 2019-era standard library, exactly
# the kind of mismatch that broke Oriented-DETR's `ms_deform_attn_cuda.cu`
# earlier in this project.
#
# **`TORCH_CUDA_ARCH_LIST` is set explicitly, from `nvidia-smi`.** Without it,
# torch's `_get_cuda_arch_flags()` calls `torch.cuda.get_device_capability()`
# itself, which fails with "Found no NVIDIA driver on your system" on a CPU
# runtime -- confirmed live, after several minutes of successful compiling. It
# was previously pinned to a literal `7.5+PTX`, on the assumption that a T4 was
# the only usable card. It is not: anything Turing or older works, and deriving
# the value here means a V100/P100/K80 runtime gets native cubins for its own
# architecture instead of PTX that cannot target it. `+PTX` is kept so the
# build is not strictly arch-locked, and the value falls back to `7.5` if the
# driver is too old to report `compute_cap`. Newer-than-Turing cards are
# rejected by the gate at the top of file setup long before this cell, so CUDA
# 10.0's nvcc is never asked for an architecture that postdates it.
#
# The clone is guarded so re-running this cell in a runtime that already has
# the repo is a no-op rather than a noisy "destination path already exists".
# !test -d /content/s2anet || git clone https://github.com/csuhan/s2anet.git /content/s2anet
# !cd /content/s2anet && \
#     ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)"; \
#     test -n "$ARCH" || ARCH=7.5; \
#     echo "compiling the s2anet ops for compute capability $ARCH"; \
#     CUDA_HOME=/content/cuda-10.0 \
#     TORCH_CUDA_ARCH_LIST="$ARCH+PTX" \
#     /content/miniconda/bin/conda run -n s2anet --no-capture-output python setup.py develop

# %%
# DOTA_devkit/polyiou -- a SWIG-wrapped C++ extension `mmdet/datasets/dota.py`
# imports at MODULE level (`from DOTA_devkit.ResultMerge_multi_process import
# mergebypoly`), so it must be importable even though this notebook never
# calls DotaDataset.evaluate() (predictions are scored through the shared
# src/obb_eval.py evaluator instead, like every other arm). Unlike
# Oriented-DETR's DOTA_devkit, this repo's csrc/ has NO pre-generated
# polyiou_wrap.cxx -- docs/INSTALL.md documents running `swig` as its own
# required step, not just a C++ compile.
#
# Always re-run swig + build_ext here rather than self-healing-checking for
# an existing build first (contrast with the Oriented-DETR notebooks' own
# self-healing cell): a full re-run only costs a few seconds, and that
# self-healing check's own loose fallback glob was the exact bug found and
# fixed in the Oriented-DETR notebooks (see CLAUDE.md) -- not worth
# reintroducing the same class of bug here to save a few seconds.
# Both commands go through `conda run -n s2anet` for the same reason the
# setup.py build above does: swig itself lives in the s2anet env's own bin/
# (never on a bare Colab cell's PATH), and this C++ compile needs the same
# activation-provided CPPFLAGS/`-I$PREFIX/include` that the crypt.h failure
# above was caused by missing.
# !cd /content/s2anet/DOTA_devkit/polyiou && /content/miniconda/bin/conda run -n s2anet --no-capture-output swig -c++ -python csrc/polyiou.i
# !cd /content/s2anet/DOTA_devkit/polyiou && /content/miniconda/bin/conda run -n s2anet --no-capture-output python setup.py build_ext --inplace

# %% [markdown]
# Mounts Drive and points `PROJECT_ROOT` at the project folder, per
# `results/colab_project_structure.md`. Same rationale as every other arm:
# training runs against local Colab disk (`PROJECT`) because Drive I/O would
# bottleneck the many small per-fold writes, and only the final archived
# results get copied to Drive at the end.

# %%
from google.colab import drive
drive.mount('/content/drive')

PROJECT_NAME = 'ProjZ'
PROJECT_ROOT = f'/content/drive/MyDrive/{PROJECT_NAME}'

# %%
# !unzip -q /content/drive/MyDrive/Krong\ Si\ Su\ Pap/datasetv3_2.zip -d /content
# !cp /content/drive/MyDrive/Krong\ Si\ Su\ Pap/classes.txt /content/classes.txt

# %%
# `unzip -q` above is quiet on purpose -- the archive is thousands of files --
# which means nothing reports a failed or skipped extraction. The first symptom
# then appears much later, in the config cell below, as
# `FileNotFoundError: /content/data/classes.txt`, which reads like a wrong-path
# bug rather than "the dataset was never staged" (confirmed live; the paths
# were correct and identical to every other arm's). So this cell verifies its
# own result instead of trusting it -- the same reason `-q` was dropped from
# the CUDA download in "file setup" above, where a silent 404 cost far more.
#
# `mv` is guarded because it FAILS if /content/data already exists (it would
# move the source inside it instead), so re-running this in a runtime that is
# already staged is now a no-op rather than a confusing error.
# !test -d /content/data || mv /content/datasetv3_2 /content/data
# !test -f /content/data/classes.txt && test -d /content/data/train/images && test -d /content/data/val/images && test -d /content/data/test/images && echo "dataset staged: OK" || echo "DATASET NOT STAGED -- run the unzip cell above and check the Drive path/filename"

# %%
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from collections import deque
from pathlib import Path
from statistics import mean

import numpy as np
import optuna
import pandas as pd

# src/ ships the pieces that MUST be identical across all five architecture
# arms -- fold assignment, E-IRFS, and the shared rotated-IoU metric. E-IRFS
# itself (EIRFSWeights/attach_eirfs) is NOT imported here -- it's applied
# (or, in this ablation twin, skipped) inside src/s2anet_worker.py, which
# runs under the s2anet conda env (see "file setup" above), not this
# notebook's own kernel. yolo_obb_to_s2anet_ann IS imported here -- it's pure
# numpy/PIL, no mmdet dependency, so it runs fine on this notebook's own
# Python 3.13 kernel.
sys.path.insert(0, PROJECT_ROOT)
from src.obb_common import (  # noqa: E402
    collect_items, load_class_names, make_folds, yolo_obb_to_s2anet_ann,
)
from src.obb_eval import (  # noqa: E402
    evaluate_obb, load_ground_truth, load_predictions, poly_iou_matrix,
    precision_recall_f1,
)
from src.obb_logging import (  # noqa: E402
    Timer, collect_epoch_log, dump_study, durable_dir, fmt_dt, mirror_file,
    open_study, write_fold_composition, write_per_class_tables, write_run_env,
)

# Ablation switch -- src/s2anet_worker.py reads this flag (via the
# "use_eirfs" key in each fold's worker_args.json) and skips attach_eirfs()
# entirely when it's False, so the dataset falls back to mmdet's own stock
# (unweighted) sampling. This is the ONLY functional difference from
# S2ANet_E-IRFS_kfold_datasetv3.py.
USE_EIRFS = False
TRAIN_EPOCHS = 250

# The s2anet conda env's own Python -- everything mmcv/mmdet dependent runs
# as a subprocess through this binary, never imported directly into this
# notebook's own Python 3.13 kernel. See src/s2anet_worker.py.
CONDA_PY = "/content/miniconda/envs/s2anet/bin/python"
WORKER_SCRIPT = f"{PROJECT_ROOT}/src/s2anet_worker.py"

# --- Edit these paths ---
DATA_ROOT   = "/content/data"
IMAGES_DIR  = f"{DATA_ROOT}/train/images"
LABELS_DIR  = f"{DATA_ROOT}/train/labels"
# K-fold carves its own held-out validation portion out of every fold, so a
# separate fixed val split serves no purpose here -- pooled into `items`
# below instead of left untouched throughout every fold (same reasoning as
# every other arm's K-fold notebook).
VAL_IMAGES  = f"{DATA_ROOT}/val/images"
VAL_LABELS  = f"{DATA_ROOT}/val/labels"
TEST_IMAGES = f"{DATA_ROOT}/test/images"
TEST_LABELS = f"{DATA_ROOT}/test/labels"
CLASSES_TXT = f"{DATA_ROOT}/classes.txt"

S2ANET_REPO = "/content/s2anet"
# S2A-Net-R50-FPN (configs/dota/s2anet_r50_fpn_1x_dota.py in the repo) is
# the paper's headline single-stage configuration -- closest capacity match
# to the other arms' R50/YOLO11m-scale backbones. Cascade variants
# (CS2A-Net) and the Swin/R101 configs in the same repo would confound
# architecture with backbone/stage-count capacity. There is no `BASE_CONFIG`
# constant here the way Strip R-CNN has one: mmcv==0.2.14 (confirmed by
# reading its actual source, mmcv/utils/config.py) predates the `_base_`
# config-inheritance mechanism entirely, so `build_config()` below writes
# every field inline, hand-matched to that stock config's values, rather
# than inheriting from it.

# --- Training/HPO knobs ---
DEVICE      = 0
# Same absolute-path requirement as every other arm's notebook: a relative
# work_dir/PROJECT gets re-rooted by the framework and splits training
# output away from this notebook's own writes. Suffixed _noeirfs so this
# twin's own output never collides with the E-IRFS notebook's PROJECT.
PROJECT     = "/content/runs/s2anet_noeirfs"
SEED             = 42
K_FOLDS          = 9
SHUFFLE_BEFORE_SPLIT = True

# Mirrors every other arm's train-only/train+val-pooled split: the embedded
# HPO search below runs against its own train-only folds (HPO_K_FOLDS), kept
# separate from the K_FOLDS train+val-pooled folds K-fold final training
# uses, so HPO and final training never see the same held-out val images.
# Same values as the E-IRFS notebook's own search -- this twin runs an
# independent search (see the header's "Deliberate deviation" note), not a
# smaller/cheaper one.
HPO_K_FOLDS      = 3
EPOCHS_PER_TRIAL = 50
N_TRIALS         = 10
N_STARTUP_TRIALS = 3

# S2A-Net (mmcv 0.2.14's EpochBasedRunner-equivalent Runner) has no
# early-stopping hook, same situation as Strip R-CNN and Oriented-DETR
# (see CLAUDE.md's "Patience/early-stopping across architectures" entry) --
# both HPO trials and final K-fold training always run their full requested
# epoch budget. No PATIENCE/FINAL_PATIENCE constant is defined for the same
# reason those two arms don't define one either.

assert Path(PROJECT).is_absolute(), f"PROJECT must be an absolute path, got {PROJECT!r}."
Path(PROJECT).mkdir(parents=True, exist_ok=True)

# Shared cache of images pre-scaled to each searched imgsz. This dataset's
# images are ~8064x6048 (48.8 MP): one decode is ~0.5 s and ~146 MB, paid per
# image per epoch, to keep 0.1 MP of it at imgsz=320 -- which measured as 46%
# of every training iteration (time 1.25 s, data_time 0.58 s). The cache holds
# each image at exactly the size RotatedResize would have produced, so the
# pipeline's own resize becomes a no-op and the network sees the same pixels;
# see resized_copy() in src/obb_common.py for the arithmetic and the proof.
# Costs ~18 min per distinct imgsz the search actually samples, once per
# runtime, and supersedes the size_cache this notebook used to thread through.
RESIZE_ROOT = "/content/resized"

# mmdet hardcodes workers_per_gpu=2 in its DOTA configs; Ultralytics defaults
# to workers=8, so the YOLO arms already read with more. Follow them, bounded
# by what the runtime actually has.
NUM_WORKERS = min(os.cpu_count() or 2, 8)
print(f"Dataloader workers: {NUM_WORKERS} (of {os.cpu_count()} CPUs)")

# Everything that has to outlive the runtime. Colab reclaims the VM between
# sittings and wipes /content, so a fold's predictions/metrics, the winning
# params, the Optuna study and the newest checkpoint are all mirrored here on
# Drive -- without which a 9-fold x TRAIN_EPOCHS run can never finish, since
# nothing survives to be resumed or reused. Derived from PROJECT's own name so
# this twin and the E-IRFS one cannot collide.
RUN_ROOT = f"{PROJECT_ROOT}/runs/{Path(PROJECT).name}"
HPO_DB = Path(PROJECT) / "hpo.db"
HPO_DB_MIRROR = Path(RUN_ROOT) / "hpo.db"
HPO_STUDY_NAME = Path(PROJECT).name

# Save (and mirror) a checkpoint this often during final K-fold training. At
# TRAIN_EPOCHS a fold is hours long, and with the old interval=total_epochs a
# disconnect at epoch 249 lost the entire fold. HPO folds keep the old
# behaviour -- see build_config's ckpt_interval.
CKPT_INTERVAL = 25

# Restrict final K-fold training to these fold numbers, e.g. [0, 1, 2], so the
# 9 independent folds can be split across parallel runtimes; None runs them
# all. Folds already finished (their fold_row.json is on Drive) are reused
# regardless, so the summary is complete once every fold has been run once,
# whichever session ran it.
FOLDS_TO_RUN = None

CLASS_NAMES = load_class_names(CLASSES_TXT)
NUM_CLASSES = len(CLASS_NAMES)
print(f"{NUM_CLASSES} classes: {CLASS_NAMES}")

# Best-effort run metadata -- platform, GPU, framework versions, git commit,
# and this arm's own control-variable constants. Never blocks training on
# failure; see src/obb_logging.py's module docstring. This kernel has no
# torch/mmcv at all (they live in the s2anet conda env) -- probing those
# would need a conda-run subprocess, which the notebook already does
# elsewhere for its own diagnostic one-liners; not duplicated here.
write_run_env(Path(PROJECT) / "run_env.json", repo_root=PROJECT_ROOT, extra={
    "arm": "S2ANet_NoEIRFS_KFold_Final", "SEED": SEED, "K_FOLDS": K_FOLDS,
    "TRAIN_EPOCHS": TRAIN_EPOCHS, "USE_EIRFS": USE_EIRFS, "PROJECT": PROJECT,
})

# %% [markdown]
# ### Config construction
#
# S2A-Net (like Strip R-CNN) is config-file-driven rather than kwarg-driven,
# so each trial's/fold's hyperparameters have to be written out as a config
# file before they can be used to train. Unlike Strip R-CNN's `_base_ = [...]`
# inheritance, this writes every field inline -- mmcv==0.2.14 predates the
# `_base_` mechanism (confirmed by reading its actual source) -- hand-matched
# field-for-field to the stock `configs/dota/s2anet_r50_fpn_1x_dota.py`
# except the dataset wiring and the searched hyperparameters below. The
# backbone, neck, and S2ANetHead architecture are left exactly as published.
#
# **Search space, and why it's smaller than the other arms':** S2A-Net's own
# augmentation pipeline (`train_pipeline` in the stock config) only offers
# `RotatedResize` + `RotatedRandomFlip` -- no HSV/mosaic/mixup/shear
# equivalents exist to search over (unlike YOLO's or Oriented-DETR's spaces).
# `lr_config.policy='step'` (not cosine) is this repo's own convention, kept
# as-is rather than switched to a cosine schedule none of its configs use.
#
# **One deliberate deviation from the stock config, for cross-arm
# comparability:** `test_cfg.score_thr` is 0.001, not the stock 0.05. That
# threshold is applied inside `multiclass_nms_rotated` during inference, i.e.
# *before* `src/s2anet_worker.py`'s own `predict_fold(score_thr=0.001)` ever
# sees a detection -- so leaving it at 0.05 would have silently capped this
# arm's recall and truncated the precision/recall curve `evaluate_obb()`
# integrates, understating its AP against the YOLO and Oriented-DETR arms,
# which both evaluate at 0.001. It changes the evaluation operating point
# only, never the architecture or the training objective.
#
# **Four more deliberate deviations from the stock config, all of them moving
# this arm TOWARD the other five rather than away:**
#
# * `max_per_img=300`, not the stock 2000. Ultralytics' validator defaults to
#   `max_det=300` and the YOLO notebooks never override it, so keeping 2000
#   here handed this arm ~6.7x more low-confidence detections to extend the
#   tail of the precision/recall curve `evaluate_obb` integrates. (It also
#   made each fold write ~185k predictions, which is slow to score.)
# * `LoadAnnotations(skip_img_without_anno=False)`, so images with no
#   instances train as background instead of being silently replaced by a
#   resampled different image. This dataset ships background-only negatives on
#   purpose (`analysis/dataset/phase4_background_negatives.py`) and the YOLO
#   arms train on them. Needs `patch_empty_gt_support()` in
#   `src/s2anet_worker.py` to be viable at all -- see its docstring.
# * `fp16 = dict(loss_scale=512.)`. Ultralytics defaults `amp=True`, so the
#   YOLO arms already train in mixed precision; fp32 here was the outlier.
#   mmdet-1.x swaps in `Fp16OptimizerHook` when this key is present, and
#   `S2ANetHead` already carries `@force_fp32` on `loss()`/`get_bboxes()`, so
#   the losses, the rotated IoU assignment and NMS all stay fp32 -- only the
#   backbone/FPN/head convolutions use the tensor cores. `init_detector` has
#   no fp16 path, so evaluation is an fp32 forward pass either way.
# * `workers_per_gpu=NUM_WORKERS` rather than mmdet's hardcoded 2, against
#   Ultralytics' default of 8.

# %%
def build_config(trial_params, fold_spec, work_dir, epochs, seed=SEED,
                 ckpt_interval=None):
    """Write an s2anet (mmdet-1.x style) config for one fold and return its path.

    ``ckpt_interval`` defaults to ``epochs`` (only the final epoch is saved, as
    before). The final K-fold loop passes ``CKPT_INTERVAL`` instead so a
    multi-hour fold can be resumed; HPO folds keep the default, since at
    EPOCHS_PER_TRIAL a lost fold costs under an hour and 30 folds' worth of
    intermediate checkpoints is real disk.
    """
    p = trial_params
    ckpt_interval = ckpt_interval or epochs
    cfg = f'''model = dict(
    type='S2ANetDetector',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        num_outs=5),
    bbox_head=dict(
        type='S2ANetHead',
        num_classes={NUM_CLASSES + 1},
        in_channels=256,
        feat_channels=256,
        stacked_convs=2,
        with_orconv=True,
        anchor_ratios=[1.0],
        anchor_strides=[8, 16, 32, 64, 128],
        anchor_scales=[4],
        target_means=[.0, .0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0, 1.0],
        loss_fam_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_fam_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        loss_odm_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_odm_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)))
train_cfg = dict(
    fam_cfg=dict(
        assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.5, neg_iou_thr=0.4, min_pos_iou=0,
                     ignore_iof_thr=-1, iou_calculator=dict(type='BboxOverlaps2D_rotated')),
        bbox_coder=dict(type='DeltaXYWHABBoxCoder', target_means=(0., 0., 0., 0., 0.),
                        target_stds=(1., 1., 1., 1., 1.), clip_border=True),
        allowed_border=-1, pos_weight=-1, debug=False),
    odm_cfg=dict(
        assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.5, neg_iou_thr=0.4, min_pos_iou=0,
                     ignore_iof_thr=-1, iou_calculator=dict(type='BboxOverlaps2D_rotated')),
        bbox_coder=dict(type='DeltaXYWHABBoxCoder', target_means=(0., 0., 0., 0., 0.),
                        target_stds=(1., 1., 1., 1., 1.), clip_border=True),
        allowed_border=-1, pos_weight=-1, debug=False))
test_cfg = dict(
    nms_pre=2000, min_bbox_size=0, score_thr=0.001,
    nms=dict(type='nms_rotated', iou_thr=0.1), max_per_img=300)

dataset_type = 'DotaDataset'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_scale = ({p["imgsz"]}, {p["imgsz"]})
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, skip_img_without_anno=False),
    dict(type='RotatedResize', img_scale=img_scale, keep_ratio=True),
    dict(type='RotatedRandomFlip', flip_ratio={p["flip_ratio"]:.6f}),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='RotatedResize', img_scale=img_scale, keep_ratio=True),
            dict(type='RotatedRandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu={p["batch"]},
    workers_per_gpu={NUM_WORKERS},
    train=dict(type=dataset_type,
              ann_file="{fold_spec["train_ann"]}",
              img_prefix="{fold_spec["train_images"]}/",
              pipeline=train_pipeline),
    val=dict(type=dataset_type,
            ann_file="{fold_spec["val_ann"]}",
            img_prefix="{fold_spec["val_images"]}/",
            pipeline=test_pipeline),
    test=dict(type=dataset_type,
             ann_file="{fold_spec["val_ann"]}",
             img_prefix="{fold_spec["val_images"]}/",
             pipeline=test_pipeline))

optimizer = dict(type='{p["optimizer"]}', lr={p["lr0"]:.8f}, weight_decay={p["weight_decay"]:.10f},
                 {p["_optimizer_extra"]})
optimizer_config = dict(grad_clip=dict(max_norm={p["clip_max_norm"]:.4f}, norm_type=2))
fp16 = dict(loss_scale=512.)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters={p["warmup_iters"]},
    warmup_ratio={p["warmup_ratio"]:.6f},
    step=[max(1, int({epochs} * {p["lr_step_frac1"]:.6f})), max(2, int({epochs} * {p["lr_step_frac2"]:.6f}))])
checkpoint_config = dict(interval={ckpt_interval})
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])

total_epochs = {epochs}
gpus = 1
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = "{work_dir}"
seed = {seed}
'''
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = work_dir / "config.py"
    cfg_path.write_text(cfg)
    return str(cfg_path)


def optimizer_extra(name, momentum):
    """mmdet-1.x optimizers take different extra kwargs by type -- same
    pattern as every other arm's own copy of this function.
    """
    if name == "SGD":
        return f"momentum={momentum:.6f}"
    return f"betas=({momentum:.6f}, 0.999)"

# %% [markdown]
# ### Train / predict one fold
#
# Runs entirely inside the `s2anet` conda env via subprocess (see "file
# setup" above and `src/s2anet_worker.py`'s own docstring) -- this
# notebook's own kernel can never import mmcv/mmdet directly (no
# `torch==1.3` wheel exists for modern Colab's Python 3.11+).

# %%
# How often a non-verbose fold echoes one of mmdet's own progress lines. The
# worker's output used to be captured whole and printed only once the fold had
# finished, which made a 50-epoch fold indistinguishable from a hung one for
# its entire duration -- the only way to tell was to open mmdet's log file by
# hand. A line a minute is enough to see life and an eta; echoing every line
# (what verbose=True still does for trial 0) is thousands of lines per fold.
HEARTBEAT_SECS = 60


# fmt_dt() is now src/obb_logging.fmt_dt (imported above) -- was previously
# defined here inline; kept byte-identical logic, now shared with every arm.

# A bare percentage on its own line, or the head of a real tqdm bar. See the
# streaming loop below for who emits thousands of these.
PROGRESS_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)%(?:\s*$|\|)")


def run_fold_via_conda(cfg_path, work_dir, train_labels, test_items, epochs, verbose=False,
                       label="", ckpt_mirror=None):
    """Train one fold and predict on `test_items`, entirely inside the
    `s2anet` conda env. Writes `work_dir/test_predictions.json` in the
    shared prediction format (read it back with `src.obb_eval.load_predictions`)
    and returns the checkpoint path.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    ckpt = work_dir / f"epoch_{epochs}.pth"
    predictions_out = work_dir / "test_predictions.json"

    args_path = work_dir / "worker_args.json"
    args_path.write_text(json.dumps({
        "project_root": PROJECT_ROOT,
        "cfg_path": str(cfg_path),
        "work_dir": str(work_dir),
        "epochs": epochs,
        "train_labels": [str(p) for p in train_labels],
        "class_names": list(CLASS_NAMES),
        "num_classes": NUM_CLASSES,
        # No "eirfs_threshold" key -- src/s2anet_worker.py only reads it when
        # use_eirfs is True, which it never is in this ablation twin.
        "use_eirfs": USE_EIRFS,
        "verbose": verbose,
        "seed": SEED,
        "test_items": [[str(img), str(lbl)] for img, lbl in test_items],
        "device": DEVICE,
        # Where src/s2anet_worker.py mirrors each new checkpoint, and where it
        # looks for one to resume from. None disables both.
        "ckpt_mirror": str(ckpt_mirror) if ckpt_mirror else None,
        "predictions_out": str(predictions_out),
    }))

    print(f"  {label}training {epochs} epochs on {len(train_labels)} images "
          f"-- progress at most every {HEARTBEAT_SECS}s", flush=True)
    t0 = time.time()
    # Popen + line iteration rather than subprocess.run(capture_output=True):
    # the latter hands back nothing until the process exits, so a running fold
    # and a deadlocked one look identical from here. stderr is merged into
    # stdout so the two interleave in their real order (mmdet writes to both)
    # and a single pipe can be drained without threads.
    proc = subprocess.Popen([CONDA_PY, WORKER_SCRIPT, str(args_path)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    # deque: only the last few hundred lines are ever used (the failure tail
    # below is capped at 6000 chars), and a fold can emit tens of thousands.
    captured, last_beat = deque(maxlen=400), 0.0
    warn_seen, warn_repeats = set(), 0
    last_decile, in_warning, warn_is_new = -1, False, False
    # Full raw stream, unthrottled -- unlike `captured` above (bounded, and
    # only ever printed on failure), this survives the whole run so mmdet's
    # own per-iteration TextLoggerHook lines are recoverable after the fact,
    # not just in the committed .ipynb cell output.
    train_log_f = open(work_dir / "train_log.txt", "w")
    for raw in proc.stdout:
        train_log_f.write(raw)
        line = raw.rstrip()

        # A download progress meter, collapsed to one line per 10%. torch 1.3's
        # torch.hub falls back to its own meter when tqdm is absent and writes
        # "\r{:.1f}%" once per 8192-byte chunk -- ~12k updates for the 98 MB
        # resnet50 weights -- and since this pipe is read in text mode, where
        # "\r" IS a line terminator, every update arrives here as its own line.
        # Deliberately not appended to `captured` either: at maxlen=400 they
        # would evict the whole failure tail.
        progress = PROGRESS_RE.match(line)
        if progress:
            decile = int(float(progress.group(1)) // 10)
            if decile != last_decile:
                last_decile = decile
                print(f"    {label}download {decile * 10}%", flush=True)
            continue

        # torch 1.3 warns on every single iteration that indexing with a uint8
        # mask is deprecated -- mmdet 1.x builds its own `valid_flags` as
        # dtype=torch.uint8, so this fires thousands of times per fold and
        # buries everything else. Collapsed to its first occurrence, and
        # collapsed GENERICALLY rather than by matching that one string: the
        # "no valid gt bbox" warning is worth seeing once too, and a denylist
        # of known-noisy strings would eventually hide something new. Printed
        # once even when not verbose, since a distinct warning is worth one
        # line; the count of what was collapsed is reported at the end.
        if "Warning" in line or "warning:" in line:
            in_warning, warn_is_new = True, line not in warn_seen
            if not warn_is_new:
                warn_repeats += 1
                continue
            warn_seen.add(line)
            captured.append(raw)
            print(f"    {label}{line}", flush=True)
            continue
        # ...and the source line Python echoes underneath each warning it
        # raises, which is indented. mmdet's own log lines all start with a
        # timestamp, so leading whitespace here can only be that echo.
        if in_warning and raw[:1].isspace():
            if not warn_is_new:
                warn_repeats += 1
                continue
            captured.append(raw)
            print(f"    {label}{line}", flush=True)
            continue
        in_warning = False

        captured.append(raw)
        if verbose:
            print(line, flush=True)
            continue
        if "Epoch [" in line:
            # mmdet's TextLoggerHook line carries the iteration counts, losses
            # and its own eta -- everything needed to judge progress. Throttled,
            # and trimmed of its leading timestamp, which Colab already shows.
            if time.time() - last_beat >= HEARTBEAT_SECS:
                last_beat = time.time()
                print(f"    {label}{line[line.index('Epoch ['):]}", flush=True)
        elif line.startswith(("Wrote ", "CHECKPOINT_PATH=")) or any(
                m in line for m in ("Traceback", "Error", "error:", "Killed",
                                    "out of memory")):
            # Never throttle anything reporting completion or trouble.
            print(f"    {label}{line}", flush=True)
    proc.wait()
    train_log_f.close()
    collect_epoch_log(work_dir, "mmdet")
    if proc.returncode != 0:
        if not verbose:        # already echoed line by line above
            print("".join(captured)[-6000:])
        raise RuntimeError(f"s2anet_worker.py failed for {work_dir} -- see output above.")
    if warn_repeats:
        print(f"  {label}({warn_repeats} repeated warning lines collapsed, "
              f"{len(warn_seen)} distinct)", flush=True)
    print(f"  {label}worker finished in {fmt_dt(time.time() - t0)}", flush=True)
    if not ckpt.exists():
        raise RuntimeError(f"no checkpoint at {ckpt} after worker run -- see output above.")
    if not predictions_out.exists():
        raise RuntimeError(f"no predictions at {predictions_out} after worker run.")
    return ckpt

# %% [markdown]
# ### Dataset materialisation
#
# The folds themselves come from `make_folds(..., seed=SEED)` -- the same
# call, with the same seed and the same sorted item list, that the other
# four arms make. `yolo_obb_to_s2anet_ann` is that transcription and nothing
# more -- it denormalises coordinates, converts polygons to the 5-value
# rotated-box representation S2A-Net's dataset pipeline expects, and writes
# them as JSON for `src/s2anet_worker.py` to turn into the pickle mmdet
# loads; it does not filter, clip, or re-order any instance. See its
# docstring in `src/obb_common.py`.
#
# Materialisation is **per `imgsz`** and therefore lazy: `imgsz` is a searched
# hyperparameter, so a fold's images can only be pre-scaled once the trial
# that needs them exists. `fold_spec_for` does that on first use and caches
# the result, so the K overlapping folds sharing an image share one scaled
# copy (`RESIZE_ROOT`). The cells below only compute the fold *assignment*.

# %%
_materialised = {}


def fold_spec_for(spec, imgsz, fold_root):
    """Materialise this fold's train/val splits at `imgsz` (once) and return a
    spec carrying the paths `build_config` needs.

    The `@{imgsz}` suffix is not cosmetic: each record's width/height are the
    SCALED ones, so a directory materialised at one imgsz is wrong at another.
    """
    key = (str(fold_root), spec["fold"], imgsz)
    if key not in _materialised:
        base = Path(fold_root) / f"fold{spec['fold']}"
        train_images, train_ann = yolo_obb_to_s2anet_ann(
            spec["train_items"], base / f"train@{imgsz}",
            resize_to=imgsz, resize_root=RESIZE_ROOT)
        val_images, val_ann = yolo_obb_to_s2anet_ann(
            spec["val_items"], base / f"val@{imgsz}",
            resize_to=imgsz, resize_root=RESIZE_ROOT)
        _materialised[key] = {
            "train_images": str(train_images), "train_ann": str(train_ann),
            "val_images": str(val_images), "val_ann": str(val_ann),
        }
    return {**spec, **_materialised[key]}


# durable_dir()/mirror_file() are now src/obb_logging's shared versions
# (imported above) -- this arm's own copies were the ones they were promoted
# from. This thin wrapper keeps every existing `durable_dir(name)` call site
# (closing over this notebook's own RUN_ROOT) unchanged.
def durable_dir(name, _shared=durable_dir):
    return _shared(RUN_ROOT, name)

# %%
# HPO fold pool: train/ only (mirrors every other arm's own train-only HPO
# split).
hpo_items = collect_items(IMAGES_DIR, LABELS_DIR)
print(f"[HPO] Found {len(hpo_items)} labeled images (train only).")
hpo_folds = make_folds(hpo_items, HPO_K_FOLDS, seed=SEED, shuffle=SHUFFLE_BEFORE_SPLIT)

HPO_FOLD_ROOT = Path(PROJECT) / "hpo_folds"
hpo_fold_specs = [{
    "fold": f["fold"],
    "train_items": f["train"],
    "val_items": f["val"],
    "train_labels": [lbl for _, lbl in f["train"]],
} for f in hpo_folds]
print(f"[HPO] {len(hpo_fold_specs)} train-only folds; images are materialised "
      f"per imgsz on first use, under {HPO_FOLD_ROOT}")

# %%
# K-fold final-training pool: train/ + val/ pooled (see the note by
# VAL_IMAGES above for why the fixed val split doesn't need to stay separate
# here).
items = collect_items(IMAGES_DIR, LABELS_DIR) + collect_items(VAL_IMAGES, VAL_LABELS)
print(f"Found {len(items)} labeled images (train + val pooled).")

folds = make_folds(items, K_FOLDS, seed=SEED, shuffle=SHUFFLE_BEFORE_SPLIT)
print("Fold sizes (train/val):", [(len(f["train"]), len(f["val"])) for f in folds])
write_fold_composition(Path(PROJECT) / "fold_composition.csv", folds, CLASS_NAMES, NUM_CLASSES)

FOLD_ROOT = Path(PROJECT) / "folds"
fold_specs = [{
    "fold": f["fold"],
    "train_items": f["train"],
    "val_items": f["val"],
    "train_labels": [lbl for _, lbl in f["train"]],
} for f in folds]
print(f"{len(fold_specs)} K-fold folds; images are materialised per imgsz on "
      f"first use, under {FOLD_ROOT}")

# %% [markdown]
# ## Precision / recall / F1 at the best-F1 confidence threshold
#
# `evaluate_obb()` (the shared AP metric) deliberately has no single
# confidence operating point -- AP integrates over the whole precision/recall
# curve. Ultralytics' own `box.mp`/`box.mr`/`box.f1` (used by the YOLO K-fold
# notebooks) DO report at one point: the confidence threshold that maximises
# mean F1 across classes. `src/obb_eval.precision_recall_f1` reproduces that
# same operating point so this arm's P/R/F1 are comparable to the other arms'.
# This notebook used to carry its own inline copy of that function (verified
# byte-identical to the module version before deleting it) -- now imported
# above instead, so per-class P/R/F1 additions land in one place for every arm.

# %% [markdown]
# ## Hyperparameters -- embedded Optuna search
#
# Runs its own search over this notebook's own train-only HPO folds, exactly
# like the E-IRFS notebook's search -- **not** loaded from
# `S2ANet_E-IRFS_kfold_datasetv3.py`'s `best_params.json` (see the header's
# "Deliberate deviation" note: by request, this twin no longer isolates
# E-IRFS as the only varying factor between the two notebooks' results).
# Every fold here trains with `USE_EIRFS = False`, so this search already
# reflects the no-E-IRFS sampling this notebook ultimately trains with.
#
# `_optimizer_extra` is re-derived from `optimizer`/`momentum` rather than
# stored either way -- `build_config` needs it, but Optuna never sees it as
# a search parameter, so it isn't part of `best_params.json`.

# %%
def suggest_params(trial):
    optimizer = trial.suggest_categorical("optimizer", ["SGD", "Adam", "AdamW"])
    momentum = trial.suggest_float("momentum", 0.7, 0.98)
    p = {
        "optimizer": optimizer,
        "lr0": trial.suggest_float("lr0", 1e-4, 1e-1, log=True),
        "momentum": momentum,
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "clip_max_norm": trial.suggest_float("clip_max_norm", 10.0, 40.0),
        "warmup_iters": trial.suggest_int("warmup_iters", 0, 1000),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 1e-3, 1e-1, log=True),
        # Fractions of total epochs the "step" LR schedule drops at -- kept
        # as two ordered fractions rather than raw epoch counts so the same
        # search space is meaningful at both EPOCHS_PER_TRIAL and
        # TRAIN_EPOCHS scales.
        "lr_step_frac1": trial.suggest_float("lr_step_frac1", 0.5, 0.8),
        "lr_step_frac2": trial.suggest_float("lr_step_frac2", 0.8, 0.95),
        "imgsz": trial.suggest_categorical("imgsz", [320, 640, 832, 1024]),
        "batch": trial.suggest_categorical("batch", [2, 4, 8]),
        "flip_ratio": trial.suggest_float("flip_ratio", 0.0, 0.9),
    }
    p["_optimizer_extra"] = optimizer_extra(optimizer, momentum)
    return p


def objective(trial):
    p = suggest_params(trial)
    fold_scores, rows, fold_errors = [], [], []
    trial_t0 = time.time()
    print(f"\n{'=' * 70}\n[Trial {trial.number}] {p['optimizer']} "
          f"lr0={p['lr0']:.2e} wd={p['weight_decay']:.1e} imgsz={p['imgsz']} "
          f"batch={p['batch']} flip={p['flip_ratio']:.2f} | {EPOCHS_PER_TRIAL} epochs "
          f"x {len(hpo_fold_specs)} folds\n{'=' * 70}", flush=True)

    for i, spec in enumerate(hpo_fold_specs, 1):
        k = spec["fold"]
        work_dir = Path(PROJECT) / f"trial_{trial.number:03d}_fold{k}"
        label = f"[T{trial.number} fold {i}/{len(hpo_fold_specs)}] "
        fold_t0 = time.time()
        try:
            # Predictions are mirrored to Drive so a finished fold is never
            # retrained after a disconnect. Checkpoints are NOT mirrored for
            # HPO folds: at EPOCHS_PER_TRIAL a lost fold costs well under an
            # hour, against ~330 MB of Drive writes per checkpoint -- the final
            # K-fold loop, where one fold is hours, does mirror them.
            durable = durable_dir(work_dir.name)
            preds_path = durable / "test_predictions.json"
            if preds_path.exists():
                print(f"  {label}already finished -- reusing {preds_path}", flush=True)
            else:
                spec = fold_spec_for(spec, p["imgsz"], HPO_FOLD_ROOT)
                cfg_path = build_config(p, spec, work_dir, EPOCHS_PER_TRIAL)
                run_fold_via_conda(cfg_path, work_dir, spec["train_labels"], spec["val_items"],
                                   EPOCHS_PER_TRIAL, verbose=(trial.number == 0), label=label)
                mirror_file(work_dir / "test_predictions.json", preds_path)
            records = load_predictions(preds_path)
            gt_fold = load_ground_truth(spec["val_items"])
            result = evaluate_obb(gt_fold, records, NUM_CLASSES, CLASS_NAMES)
            prf = precision_recall_f1(gt_fold, records, NUM_CLASSES)
            score = float(result["mAP50-95"])
            if not np.isfinite(score):
                raise RuntimeError(f"shared evaluator returned {score} for fold {k}")
        except Exception as e:
            print(f"\n{'!'*70}\n[Trial {trial.number}] Fold {k} FAILED: {e}\n{'!'*70}")
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()
            fold_errors.append(f"fold {k}: {type(e).__name__}: {e}")
            rows.append({
                "trial": trial.number, "fold": k,
                "mAP50-95": None, "mAP50": None, "mAP75": None,
                "precision": None, "recall": None, "F1": None,
                "error": str(e),
            })
            continue

        fold_scores.append(score)
        print(f"  {label}mAP50-95={score:.4f} mAP50={result['mAP50']:.4f} "
              f"F1={prf['F1']:.4f} | trial mean so far {mean(fold_scores):.4f} "
              f"| fold took {fmt_dt(time.time() - fold_t0)}", flush=True)
        rows.append({
            "trial": trial.number, "fold": k,
            "mAP50-95": score, "mAP50": result["mAP50"], "mAP75": result["mAP75"],
            "precision": prf["precision"], "recall": prf["recall"], "F1": prf["F1"],
            "error": None,
        })
        trial.report(mean(fold_scores), step=k + 1)
        # trial.report() alone prunes nothing -- Optuna requires the objective
        # to check should_prune() and raise TrialPruned itself. Without this
        # the PercentilePruner configured in run_hpo() (and every report()
        # call above) is dead weight: every trial runs every remaining fold to
        # completion no matter how bad an early fold was. Same omission that
        # was found and fixed in the Strip R-CNN notebook.
        if trial.should_prune():
            pd.DataFrame(rows).to_csv(Path(PROJECT) / f"cv_trial_{trial.number:03d}.csv", index=False)
            raise optuna.TrialPruned(
                f"pruned after fold {k} (intermediate mean mAP50-95={mean(fold_scores):.4f})"
            )

    pd.DataFrame(rows).to_csv(Path(PROJECT) / f"cv_trial_{trial.number:03d}.csv", index=False)

    if not fold_scores:
        detail = "; ".join(fold_errors)
        if trial.number == 0:
            raise RuntimeError(
                f"Trial 0 failed on all {len(hpo_fold_specs)} folds -- aborting the study. "
                f"Causes: {detail}"
            )
        raise optuna.TrialPruned(f"all folds failed: {detail}")

    if fold_errors:
        print(f"[Trial {trial.number}] WARNING: {len(fold_errors)}/{len(hpo_fold_specs)} folds failed.")
    print(f"[Trial {trial.number}] done in {fmt_dt(time.time() - trial_t0)} -- mean "
          f"mAP50-95 {mean(fold_scores):.4f} over {len(fold_scores)} fold(s)", flush=True)
    return float(mean(fold_scores))


def hpo_progress(study, trial):
    """Optuna callback: one line per finished trial with study-level progress.

    Optuna's own INFO log already reports each trial's value, params and the
    best so far, so this deliberately adds only what it does not: how many of
    N_TRIALS are done, and an eta from the mean duration of the trials that
    have finished. Without it a study that prunes quietly gives no sense of how
    much is left.
    """
    finished = [t for t in study.trials if t.state.is_finished()]
    durations = [t.duration.total_seconds() for t in finished if t.duration is not None]
    try:
        best = f"{study.best_value:.4f} (trial {study.best_trial.number})"
    except ValueError:
        best = "none yet"      # raised while nothing has COMPLETEd
    value = "n/a" if trial.value is None else f"{trial.value:.4f}"
    remaining = max(N_TRIALS - len(finished), 0)
    eta = (f" | eta {fmt_dt(mean(durations) * remaining)}"
           if durations and remaining else "")
    print(f"[HPO] trial {trial.number} {trial.state.name} value={value} | best {best} "
          f"| {len(finished)}/{N_TRIALS} finished{eta}", flush=True)
    # Once per finished trial is the natural point to push the study to Drive:
    # it is the only moment its contents changed, and the file is tiny.
    if HPO_DB.exists():
        mirror_file(HPO_DB, HPO_DB_MIRROR)


def run_hpo() -> dict:
    sampler = optuna.samplers.TPESampler(seed=SEED, n_startup_trials=N_STARTUP_TRIALS)
    pruner = optuna.pruners.PercentilePruner(
        percentile=50, n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=2
    )
    # The study lives in SQLite so an interrupted search resumes instead of
    # starting over: with no storage it is in-memory, and a disconnect loses
    # every completed trial. Kept on local disk and mirrored to Drive after each
    # trial (see hpo_progress) rather than opened directly on Drive, because
    # SQLite over a FUSE mount is a known way to corrupt a database. This is
    # the pattern src/obb_logging.open_study() promotes -- same restore-from-
    # mirror-if-local-is-gone logic, now shared with every other arm.
    study = open_study(HPO_STUDY_NAME, HPO_DB, HPO_DB_MIRROR, sampler, pruner)
    # n_trials is how many MORE trials to run, so a resumed study has to
    # subtract what it already finished or the search silently grows past
    # N_TRIALS. (A resumed search is not bit-reproducible from SEED either --
    # TPESampler's own RNG state is not persisted with the study.)
    done = len([t for t in study.trials if t.state.is_finished()])
    if done:
        print(f"[HPO] {done}/{N_TRIALS} trials already finished in this study")
    if done < N_TRIALS:
        study.optimize(objective, n_trials=N_TRIALS - done, callbacks=[hpo_progress])

    completed = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
    print(f"Completed trials: {len(completed)} / {N_TRIALS} "
          f"(pruned/failed: {N_TRIALS - len(completed)})")
    if not completed:
        raise RuntimeError("No trial completed successfully; best_params.json NOT written.")
    distinct_values = {round(t.value, 10) for t in completed if t.value is not None}
    if len(distinct_values) == 1 and len(completed) > 1:
        raise RuntimeError(
            f"All {len(completed)} completed trials returned the identical value "
            f"{distinct_values.pop()} -- trial 0 would win by tie-break regardless of "
            f"the search. best_params.json NOT written."
        )

    print("Best CV mAP50-95:", study.best_value)
    print("Best params:", study.best_params)

    final_dir = Path(PROJECT) / "best_final_s2anet_noeirfs"
    final_dir.mkdir(parents=True, exist_ok=True)
    (final_dir / "best_params.json").write_text(json.dumps(study.best_params, indent=2))
    mirror_file(final_dir / "best_params.json", Path(RUN_ROOT) / "best_params.json")
    dump_study(study, final_dir)
    mirror_file(final_dir / "hpo_trials.csv", Path(RUN_ROOT) / "hpo_trials.csv")
    params = dict(study.best_params)
    params["_optimizer_extra"] = optimizer_extra(params["optimizer"], params["momentum"])
    return params


# %% [markdown]
# ### Smoke test: one epoch on three images, two of them label-free
#
# ~1 minute, and the cheapest place to discover that one of this run's
# load-bearing pieces is broken -- before the search spends tens of hours. It
# exercises end to end: fp16 (whether the custom rotated CUDA ops have half
# kernels at all, and whether the static `loss_scale` overflows to `nan` on the
# first iteration), the JSON->pickle annotation hand-off including the empty
# `(0, 5)` case, `LoadAnnotations` no longer skipping instance-free images,
# `RotatedResize` on empty boxes, the patched assigner, the focal loss with
# zero positives, a batch mixing background-only and annotated images, the
# per-imgsz resize cache, and the checkpoint mirror that resume depends on.
# Same reasoning as the GPU-capability gates in "file setup": put a cheap probe
# beside every expensive step.
#
# On `not implemented for 'Half'` or a `nan`/`inf` loss, drop the
# `fp16 = dict(loss_scale=512.)` line from `build_config` (or lower the scale)
# and re-run -- nothing else here depends on it. Delete neither the empty-label
# images nor `skip_img_without_anno=False`: those are the correctness fix, not
# an optimisation.

# %%
SELFTEST_DIR = Path(PROJECT) / "selftest"

_st_all = hpo_folds[0]["train"]
_st_ann = next((it for it in _st_all if Path(it[1]).stat().st_size > 0), None)
assert _st_ann is not None, "no annotated training image found for the smoke test"
_st_rest = [it for it in _st_all if it[0] != _st_ann[0]][:2]
assert len(_st_rest) == 2, "need 3 distinct training images for the smoke test"

# Two of the three get blanked label files, so one batch carries both
# background-only and annotated images -- the combination that broke before.
_st_labels = SELFTEST_DIR / "labels"
_st_labels.mkdir(parents=True, exist_ok=True)
_st_pairs = [_st_ann]
for _img, _ in _st_rest:
    _blank = _st_labels / f"{Path(_img).stem}.txt"
    _blank.write_text("")
    _st_pairs.append((_img, _blank))

# Fixed hyperparameters, not sampled ones: this is testing the plumbing, and it
# should behave the same however the search happens to start.
_st_p = {
    "optimizer": "SGD", "lr0": 0.001, "momentum": 0.9, "weight_decay": 1e-4,
    "clip_max_norm": 35.0, "warmup_iters": 0, "warmup_ratio": 0.1,
    "lr_step_frac1": 0.6, "lr_step_frac2": 0.9, "imgsz": 320, "batch": 2,
    "flip_ratio": 0.5,
}
_st_p["_optimizer_extra"] = optimizer_extra(_st_p["optimizer"], _st_p["momentum"])
_st_spec = fold_spec_for(
    {"fold": "selftest", "train_items": _st_pairs, "val_items": _st_pairs,
     "train_labels": [lbl for _, lbl in _st_pairs]},
    _st_p["imgsz"], SELFTEST_DIR)
_st_work = SELFTEST_DIR / "run"
_st_cfg = build_config(_st_p, _st_spec, _st_work, 1, ckpt_interval=1)
run_fold_via_conda(_st_cfg, _st_work, _st_spec["train_labels"], _st_pairs, 1,
                   verbose=True, label="[selftest] ",
                   ckpt_mirror=SELFTEST_DIR / "mirror")

_st_preds = json.loads((_st_work / "test_predictions.json").read_text())["predictions"]
assert (SELFTEST_DIR / "mirror" / "epoch_1.pth").exists(), (
    "the checkpoint mirror never received epoch_1.pth -- resume would silently "
    "not work, and a disconnected fold would restart from scratch")
print(f"\nSMOKE TEST PASSED: 1 epoch on 3 images (2 of them label-free), "
      f"{len(_st_preds)} predictions written, checkpoint mirrored to Drive.")

# %%
best_params = run_hpo()
print("Using best_params:", json.dumps(
    {k: v for k, v in best_params.items() if k != "_optimizer_extra"}, indent=2))

# %% [markdown]
# ## K-fold final training
#
# Trains K models -- one per fold above -- each for the full `TRAIN_EPOCHS`
# using `best_params`, then evaluates every fold's model on the same
# held-out test set through the shared `src/obb_eval.py` evaluator. The fold
# with the best test mAP50-95 is kept as the representative "best" model for
# downstream use.
#
# **This phase is the bulk of the runtime and is designed to be interrupted.**
# Every fold's row, metrics and predictions are written under `RUN_ROOT` on
# Drive as it finishes, and a checkpoint is mirrored there every
# `CKPT_INTERVAL` epochs, so:
#
# * Re-running this cell retrains nothing that already finished -- finished
#   folds are read back from Drive, wherever they were run.
# * A fold interrupted part-way resumes from its last mirrored checkpoint
#   (epoch, iteration and optimizer state, not just weights) rather than
#   restarting. Without this, a disconnect at epoch 249 of 250 lost the fold.
# * The K folds are independent, so `FOLDS_TO_RUN = [0, 1, 2]` in one runtime
#   and `[3, 4, 5]` in another splits the wall-clock N ways. The summary below
#   is complete once every fold has been run once, by whichever session.
#
# The summary cell says so explicitly when folds are still missing, rather
# than quietly reporting a mean over however many happen to be present.

# %%
KFOLD_FINAL_DIR = Path(PROJECT) / "kfold_final_s2anet_noeirfs"
KFOLD_FINAL_DIR.mkdir(parents=True, exist_ok=True)

test_items = collect_items(TEST_IMAGES, TEST_LABELS)
print(f"Test images: {len(test_items)}")
gt = load_ground_truth(test_items)

fold_rows = []
fold_errors = []
per_fold_per_class = []
for spec in fold_specs:
    k = spec["fold"]
    work_dir = KFOLD_FINAL_DIR / f"fold{k}"
    # A fold that finished in ANY session is reused from Drive rather than
    # retrained -- this is what lets 9 x TRAIN_EPOCHS be spread over several
    # sittings (and across parallel runtimes, via FOLDS_TO_RUN) and still
    # produce one complete summary. Reused before the FOLDS_TO_RUN filter, so
    # a runtime told to train only some folds still reports all the finished
    # ones.
    durable = durable_dir(f"fold{k}")
    row_path = durable / "fold_row.json"
    if row_path.exists():
        fold_rows.append(json.loads(row_path.read_text()))
        reused_metrics_path = durable / "test_metrics.json"
        if reused_metrics_path.exists():
            per_fold_per_class.append(
                (k, json.loads(reused_metrics_path.read_text()).get("per_class", {})))
        print(f"[Fold {k}] already finished -- reusing {row_path}")
        continue
    if FOLDS_TO_RUN is not None and k not in FOLDS_TO_RUN:
        print(f"[Fold {k}] not in FOLDS_TO_RUN -- skipped")
        continue
    # Assigned before the try so the failure path below can report it even when
    # materialisation or config-writing is what raised.
    cfg_path = None
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        spec = fold_spec_for(spec, best_params["imgsz"], FOLD_ROOT)
        cfg_path = build_config(best_params, spec, work_dir, TRAIN_EPOCHS,
                                ckpt_interval=CKPT_INTERVAL)
        with Timer() as train_timer:
            ckpt = run_fold_via_conda(cfg_path, work_dir, spec["train_labels"], test_items,
                                      TRAIN_EPOCHS, verbose=(k == 0), ckpt_mirror=durable)
        # Mirror everything /content won't survive to Drive: the resolved
        # config, and the loss curve (previously the one place a *loss* was
        # lost outright rather than merely unparsed, since this arm's own
        # sessions are explicitly designed to be interrupted and resumed).
        durable_cfg_path = mirror_file(cfg_path, durable / "config.py")
        for log_name in ("train_log.txt", "epoch_log.csv", "epoch_log_summary.json"):
            log_src = work_dir / log_name
            if log_src.exists():
                mirror_file(log_src, durable / log_name)
        (work_dir / "resolved_params.json").write_text(json.dumps(
            {k_: v for k_, v in best_params.items() if k_ != "_optimizer_extra"}, indent=2))
        mirror_file(work_dir / "resolved_params.json", durable / "resolved_params.json")

        with Timer() as eval_timer:
            records = load_predictions(work_dir / "test_predictions.json")
            metrics = evaluate_obb(gt, records, NUM_CLASSES, CLASS_NAMES)
            prf = precision_recall_f1(gt, records, NUM_CLASSES, class_names=CLASS_NAMES)
        metrics["precision"] = prf["precision"]
        metrics["recall"] = prf["recall"]
        metrics["F1"] = prf["F1"]
        for _name, _pc in prf.get("per_class", {}).items():
            metrics["per_class"].setdefault(_name, {}).update(_pc)
        for _name, _c in prf.get("counts", {}).items():
            metrics["per_class"].setdefault(_name, {}).update(_c)
        per_fold_per_class.append((k, metrics["per_class"]))
        (work_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
        (durable / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
        mirror_file(work_dir / "test_predictions.json", durable / "test_predictions.json")
    except Exception as e:
        # One fold failing (a transient CUDA OOM, a conda-subprocess hiccup)
        # must not take down the whole 9-fold x TRAIN_EPOCHS run -- earlier
        # folds' checkpoints/test_metrics.json are already on disk, and the
        # CSV/summary should still be written from whatever folds succeeded.
        # Mirrors every other arm's K-fold final loop; the NaN-tolerant check
        # right below this loop already assumed this existed.
        print(f"\n{'!'*70}\n[Fold {k}] FAILED: {e}\n{'!'*70}")
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        fold_errors.append(f"fold {k}: {type(e).__name__}: {e}")
        fold_rows.append({
            "fold": k, "test_mAP50-95": None, "test_mAP50": None, "test_mAP75": None,
            "test_precision": None, "test_recall": None, "test_F1": None,
            "ckpt": None, "cfg_path": cfg_path, "run_dir": str(work_dir), "error": str(e),
            "started_at": started_at, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "train_seconds": None, "eval_seconds": None,
            "epochs_completed": None, "final_train_loss": None,
        })
        continue

    # `ckpt`/`run_dir` point at the Drive-side copies, not /content: a later
    # session reads these back (the test cell loads run_dir/test_metrics.json)
    # long after the local run directory has been wiped. The mirror holds the
    # final epoch because patch_checkpoint_mirror keeps the newest, which at
    # the end of training is epoch_TRAIN_EPOCHS. `cfg_path` now points at that
    # same mirrored copy for the same reason -- the local work_dir it used to
    # point at dangles once the session ends.
    durable_ckpt = durable / f"epoch_{TRAIN_EPOCHS}.pth"
    epoch_summary_path = durable / "epoch_log_summary.json"
    epoch_summary = (json.loads(epoch_summary_path.read_text())
                     if epoch_summary_path.exists() else {})
    row = {
        "fold": k,
        "test_mAP50-95": metrics["mAP50-95"],
        "test_mAP50": metrics["mAP50"],
        "test_mAP75": metrics["mAP75"],
        "test_precision": prf["precision"],
        "test_recall": prf["recall"],
        "test_F1": prf["F1"],
        "ckpt": str(durable_ckpt if durable_ckpt.exists() else ckpt),
        "cfg_path": str(durable_cfg_path),
        "run_dir": str(durable),
        "local_run_dir": str(work_dir),
        "error": None,
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "train_seconds": train_timer.seconds,
        "eval_seconds": eval_timer.seconds,
        "epochs_completed": epoch_summary.get("epochs_completed"),
        "final_train_loss": epoch_summary.get("final_loss"),
    }
    # Written only on success, so a failed fold is retried next session.
    row_path.write_text(json.dumps(row))
    fold_rows.append(row)
    print(f"[Fold {k}] test mAP50-95={metrics['mAP50-95']:.4f}  mAP50={metrics['mAP50']:.4f}  "
          f"P={prf['precision']:.4f}  R={prf['recall']:.4f}  F1={prf['F1']:.4f}"
          f"  | train {fmt_dt(train_timer.seconds or 0)}")

if fold_errors:
    print(f"WARNING: {len(fold_errors)}/{len(fold_specs)} folds failed: {'; '.join(fold_errors)}")

fold_df = pd.DataFrame(fold_rows)
if fold_df.empty:
    raise RuntimeError(
        "No fold results at all -- nothing was trained in this session and "
        "nothing was found under RUN_ROOT from an earlier one. If FOLDS_TO_RUN "
        "is set, it names folds that have neither run nor finished."
    )
fold_df = fold_df.sort_values("fold").reset_index(drop=True)
fold_df.to_csv(KFOLD_FINAL_DIR / "kfold_final_results.csv", index=False)
mirror_file(KFOLD_FINAL_DIR / "kfold_final_results.csv",
            Path(RUN_ROOT) / "kfold_final_results.csv")
write_per_class_tables(KFOLD_FINAL_DIR, per_fold_per_class, CLASS_NAMES)
for _pc_name in ("kfold_per_class_results.csv", "kfold_per_class_summary.csv"):
    if (KFOLD_FINAL_DIR / _pc_name).exists():
        mirror_file(KFOLD_FINAL_DIR / _pc_name, Path(RUN_ROOT) / _pc_name)
if len(fold_df) < len(fold_specs):
    print(f"NOTE: {len(fold_df)}/{len(fold_specs)} folds present. The summary "
          f"below is over those only -- run the remaining folds (FOLDS_TO_RUN) "
          f"and re-run this cell to complete it.")
print(fold_df)

# %%
if fold_df["test_mAP50-95"].isna().all():
    raise RuntimeError(
        "Every fold's test mAP50-95 came back NaN -- scroll up for per-fold "
        "training/eval errors. kfold_final_summary.json NOT written."
    )

summary = {
    "n_folds": int(len(fold_df)),
    # Folds that raised and were skipped. The mean/std below are over the
    # SURVIVING folds only, so n_folds alone overstates the sample.
    "n_folds_failed": int(len(fold_errors)),
    "test_mAP50-95_mean": float(fold_df["test_mAP50-95"].mean()),
    "test_mAP50-95_std": float(fold_df["test_mAP50-95"].std(ddof=0)),
    "test_mAP50_mean": float(fold_df["test_mAP50"].mean()),
    "test_mAP50_std": float(fold_df["test_mAP50"].std(ddof=0)),
    "test_mAP75_mean": float(fold_df["test_mAP75"].mean()),
    "test_mAP75_std": float(fold_df["test_mAP75"].std(ddof=0)),
    "test_precision_mean": float(fold_df["test_precision"].mean()),
    "test_precision_std": float(fold_df["test_precision"].std(ddof=0)),
    "test_recall_mean": float(fold_df["test_recall"].mean()),
    "test_recall_std": float(fold_df["test_recall"].std(ddof=0)),
    "test_F1_mean": float(fold_df["test_F1"].mean()),
    "test_F1_std": float(fold_df["test_F1"].std(ddof=0)),
    "best_fold": int(fold_df.loc[fold_df["test_mAP50-95"].idxmax(), "fold"]),
    "best_params": {k: v for k, v in best_params.items() if k != "_optimizer_extra"},
}
print("K-fold final test summary:", json.dumps(summary, indent=2))
(KFOLD_FINAL_DIR / "kfold_final_summary.json").write_text(json.dumps(summary, indent=2))
mirror_file(KFOLD_FINAL_DIR / "kfold_final_summary.json",
            Path(RUN_ROOT) / "kfold_final_summary.json")

best_row = fold_df.loc[fold_df["test_mAP50-95"].idxmax()]
FINAL_CKPT = Path(best_row["ckpt"])
FINAL_DIR = Path(best_row["run_dir"])
print(f"Best fold: {summary['best_fold']} -> {FINAL_CKPT}")

# %% [markdown]
# ### test
#
# Re-prints the held-out test metrics for the best fold (already computed
# during the K-fold loop above) plus the full K-fold summary.

# %%
metrics = json.loads((FINAL_DIR / "test_metrics.json").read_text())

print(f"\n=== S2A-Net R50-FPN | held-out test, best fold {summary['best_fold']} (shared evaluator) ===")
print(f"mAP50-95: {metrics['mAP50-95']:.4f}")
print(f"mAP50   : {metrics['mAP50']:.4f}")
print(f"mAP75   : {metrics['mAP75']:.4f}")
print(f"precision={metrics.get('precision', float('nan')):.4f}  "
      f"recall={metrics.get('recall', float('nan')):.4f}  "
      f"F1={metrics.get('F1', float('nan')):.4f}")
for name, m in metrics["per_class"].items():
    print(f"  {name:15} AP50-95={m['AP50-95']:.4f}  AP50={m['AP50']:.4f}  n={m['n_gt']}")

print("\nFull K-fold summary (all folds):", json.dumps(summary, indent=2))

# %% [markdown]
# ### Archive to Drive
#
# For "re-verify a previously archived run without retraining," see the
# dedicated `notebooks/S2ANet_E-IRFS_retest_datasetv3.py` instead of a
# section in this file -- see this notebook's header for why.

# %%
def _unique_drive_dir(base_dir: Path) -> Path:
    """Avoid merging into a stale previous archive: if base_dir already
    exists on Drive, return the first base_dir2, base_dir3, ... that doesn't."""
    if not base_dir.exists():
        return base_dir
    n = 2
    while True:
        candidate = base_dir.with_name(f"{base_dir.name}{n}")
        if not candidate.exists():
            return candidate
        n += 1


# Archives the ENTIRE local /content/runs tree, not just PROJECT -- each
# notebook gets its own Colab runtime, so its /content/runs only ever holds
# this notebook's own output -- into a notebook-labeled subfolder on Drive.
RUN_LABEL = "S2ANet_NoEIRFS_KFold_Final"

LOCAL_RUNS = Path("/content/runs")
DRIVE_DEST = _unique_drive_dir(Path(f"{PROJECT_ROOT}/runs/{RUN_LABEL}"))
DRIVE_DEST.mkdir(parents=True, exist_ok=True)
if DRIVE_DEST.name != RUN_LABEL:
    print(f"NOTE: '{RUN_LABEL}' already exists on Drive -- archiving to '{DRIVE_DEST.name}' instead.")
# The materialised fold directories are symlinked copies of the dataset and
# would either bloat the archive or copy as broken links; the fold
# *assignment* is reproducible from make_folds(items, K_FOLDS, seed=SEED)
# anyway.
shutil.copytree(LOCAL_RUNS, DRIVE_DEST, dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("folds", "hpo_folds", "selftest"))

print("Local  :", LOCAL_RUNS, "->", sorted(p.name for p in LOCAL_RUNS.iterdir()))
print("Drive  :", DRIVE_DEST, "->", sorted(p.name for p in DRIVE_DEST.iterdir()))
