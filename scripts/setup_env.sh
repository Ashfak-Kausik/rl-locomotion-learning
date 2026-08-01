#!/usr/bin/env bash
# ============================================================================
# rl-locomotion-learning — one-shot environment bootstrap
# ============================================================================
# Detects what this machine is missing and installs it:
#
#   1. Native OS packages   (apt / dnf / pacman / zypper / brew — auto-detected)
#   2. A project virtualenv (.venv) with a matching Python
#   3. All Python packages  (PyTorch flavour chosen for your GPU — see --gpu)
#   4. Verifies the result  (delegates to scripts/check_env.py)
#
# Usage:
#   ./scripts/setup_env.sh                 # check, prompt, then install
#   ./scripts/setup_env.sh --yes           # non-interactive (CI / Docker)
#   ./scripts/setup_env.sh --check-only    # report only, change nothing
#   ./scripts/setup_env.sh --gpu auto      # detect the GPU, pick the wheel
#   ./scripts/setup_env.sh --cuda          # NVIDIA CUDA PyTorch
#   ./scripts/setup_env.sh --xpu           # Intel Arc (XPU) PyTorch
#   ./scripts/setup_env.sh --rocm          # AMD ROCm PyTorch
#   ./scripts/setup_env.sh --no-system     # skip OS packages (no sudo)
#   ./scripts/setup_env.sh --python 3.12   # pick the interpreter
#
# The default is CPU-only PyTorch (~200 MB vs ~2.5 GB) because Stages 1-2 are
# CPU-only by design. Only Stage 3 training wants a GPU.
#
# Safe to re-run: every step is idempotent.
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"

# Pinned index URLs. `--cuda` used to run a bare `pip install torch>=2.4,<3`
# with NO index-url, which silently gave you whatever CUDA build PyPI happened
# to default to — not necessarily one matching your driver, and not the cu130
# build this project is verified against.
TORCH_CPU_INDEX="https://download.pytorch.org/whl/cpu"
TORCH_CUDA_INDEX="https://download.pytorch.org/whl/cu130"
TORCH_XPU_INDEX="https://download.pytorch.org/whl/xpu"
TORCH_ROCM_INDEX="https://download.pytorch.org/whl/rocm6.2"

ASSUME_YES=0
CHECK_ONLY=0
TORCH_FLAVOUR="cpu"        # cpu | cuda | xpu | rocm | auto
SKIP_SYSTEM=0
PYTHON_BIN=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --yes|-y)      ASSUME_YES=1 ;;
    --check-only)  CHECK_ONLY=1 ;;
    --cuda)        TORCH_FLAVOUR="cuda" ;;
    --xpu|--arc)   TORCH_FLAVOUR="xpu" ;;
    --rocm)        TORCH_FLAVOUR="rocm" ;;
    --gpu)         TORCH_FLAVOUR="$2"; shift ;;
    --no-system)   SKIP_SYSTEM=1 ;;
    --python)      PYTHON_BIN="python$2"; shift ;;
    -h|--help)     sed -n '2,31p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
  esac
  shift
done

case "$TORCH_FLAVOUR" in
  cpu|cuda|xpu|rocm|auto|none) ;;
  *) echo "unknown --gpu value: $TORCH_FLAVOUR (cpu|cuda|xpu|rocm|auto)" >&2
     exit 2 ;;
esac

# --- pretty output ----------------------------------------------------------
if [[ -t 1 ]]; then
  B=$'\033[1m'; G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; D=$'\033[2m'; N=$'\033[0m'
else
  B=""; G=""; Y=""; R=""; D=""; N=""
fi
step() { printf '\n%s\n%s\n' "${B}==> $*${N}" "${D}$(printf '%.0s-' {1..70})${N}"; }
ok()   { printf '  %s✓%s %s\n' "$G" "$N" "$*"; }
warn() { printf '  %s!%s %s\n' "$Y" "$N" "$*"; }
die()  { printf '  %s✗%s %s\n' "$R" "$N" "$*" >&2; exit 1; }
info() { printf '    %s%s%s\n' "$D" "$*" "$N"; }

confirm() {
  [[ $ASSUME_YES -eq 1 ]] && return 0
  read -r -p "  ${B}$1${N} [Y/n] " reply
  [[ -z "$reply" || "$reply" =~ ^[Yy] ]]
}

# ============================================================================
step "1/5  Detecting platform"
# ============================================================================
OS="$(uname -s)"
PKG=""
case "$OS" in
  Linux)
    if   command -v apt-get >/dev/null; then PKG=apt
    elif command -v dnf     >/dev/null; then PKG=dnf
    elif command -v pacman  >/dev/null; then PKG=pacman
    elif command -v zypper  >/dev/null; then PKG=zypper
    fi
    DISTRO="$( . /etc/os-release 2>/dev/null && echo "${PRETTY_NAME:-linux}" )"
    ;;
  Darwin)
    command -v brew >/dev/null && PKG=brew
    DISTRO="macOS $(sw_vers -productVersion 2>/dev/null || echo '?')"
    ;;
  *) DISTRO="$OS" ;;
esac
ok "OS: ${DISTRO}"
[[ -n "$PKG" ]] && ok "package manager: ${PKG}" || warn "no supported package manager found — native libs must be installed by hand"

SUDO=""
if [[ $EUID -ne 0 ]] && command -v sudo >/dev/null; then SUDO="sudo"; fi

# --- GPU detection ----------------------------------------------------------
# Resolves --gpu auto to a real wheel flavour. Kept in shell (rather than
# deferring to scripts/hw_profile.py) because this runs BEFORE python packages
# exist, so it cannot import torch or anything else.
detect_gpu_flavour() {
  if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
    echo cuda; return
  fi
  if command -v rocm-smi >/dev/null; then echo rocm; return; fi
  if command -v lspci >/dev/null; then
    # Intel Arc discrete only. Integrated UHD/Iris is detected by the same
    # pattern but is not worth a 2 GB wheel for Stage 3 training.
    if lspci 2>/dev/null | grep -Ei 'VGA|3D controller|Display' \
         | grep -qiE 'Intel.*(Arc|DG2|Battlemage|Alchemist)'; then
      echo xpu; return
    fi
    if lspci 2>/dev/null | grep -Ei 'VGA|3D controller' | grep -qi NVIDIA; then
      echo cuda; return
    fi
  fi
  if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
    echo mps; return
  fi
  echo cpu
}

if [[ "$TORCH_FLAVOUR" == "auto" ]]; then
  TORCH_FLAVOUR="$(detect_gpu_flavour)"
  # mps ships in the standard macOS wheel; no special index needed.
  [[ "$TORCH_FLAVOUR" == "mps" ]] && TORCH_FLAVOUR="cpu"
  ok "detected GPU flavour: ${TORCH_FLAVOUR}"
fi

# ============================================================================
step "2/5  Native OS packages"
# ============================================================================
# Why each one is needed:
#   libgl1 / libglfw3       MuJoCo viewer window + OpenGL rendering
#   libegl1 / libosmesa6    headless rendering (MUJOCO_GL=egl|osmesa)
#   build-essential/gcc     compiles Box2D (LunarLander-v3) from source
#   python3-dev             Python C headers for the same
#   swig                    generates the Box2D bindings
#   ffmpeg                  encodes the demo GIFs in media/
#   git                     version control
declare -a NEED=()

apt_pkgs=(libgl1 libglfw3 libegl1 libosmesa6 build-essential python3-dev
          python3-venv python3-pip swig ffmpeg patchelf git pkg-config)
dnf_pkgs=(mesa-libGL glfw mesa-libEGL mesa-libOSMesa gcc gcc-c++ make
          python3-devel python3-pip swig ffmpeg patchelf git pkgconf)
pacman_pkgs=(libgl glfw mesa base-devel python-pip swig ffmpeg patchelf git pkgconf)
zypper_pkgs=(Mesa-libGL1 libglfw3 Mesa-libEGL1 python3-devel python3-pip
             gcc gcc-c++ make swig ffmpeg patchelf git pkg-config)
brew_pkgs=(glfw swig ffmpeg git)

case "$PKG" in
  apt)    PKGS=("${apt_pkgs[@]}") ;;
  dnf)    PKGS=("${dnf_pkgs[@]}") ;;
  pacman) PKGS=("${pacman_pkgs[@]}") ;;
  zypper) PKGS=("${zypper_pkgs[@]}") ;;
  brew)   PKGS=("${brew_pkgs[@]}") ;;
  *)      PKGS=() ;;
esac

pkg_installed() {
  case "$PKG" in
    apt)    dpkg-query -W -f='${Status}' "$1" 2>/dev/null | grep -q "ok installed" ;;
    dnf)    rpm -q "$1" >/dev/null 2>&1 ;;
    pacman) pacman -Qi "$1" >/dev/null 2>&1 ;;
    zypper) rpm -q "$1" >/dev/null 2>&1 ;;
    brew)   brew list --formula "$1" >/dev/null 2>&1 ;;
    *)      return 1 ;;
  esac
}

if [[ ${#PKGS[@]} -eq 0 ]]; then
  warn "skipping native packages (unsupported platform)"
  info "see docs/DEPENDENCIES.md for the manual list"
else
  for p in "${PKGS[@]}"; do
    if pkg_installed "$p"; then ok "$p"; else warn "$p  ${D}(missing)${N}"; NEED+=("$p"); fi
  done
fi

if [[ ${#NEED[@]} -gt 0 ]]; then
  if [[ $CHECK_ONLY -eq 1 ]]; then
    warn "would install: ${NEED[*]}"
  elif [[ $SKIP_SYSTEM -eq 1 ]]; then
    warn "--no-system given; skipping install of: ${NEED[*]}"
  elif confirm "Install ${#NEED[@]} missing native package(s)?"; then
    case "$PKG" in
      apt)    $SUDO apt-get update -qq && $SUDO apt-get install -y "${NEED[@]}" ;;
      dnf)    $SUDO dnf install -y "${NEED[@]}" ;;
      pacman) $SUDO pacman -Sy --needed --noconfirm "${NEED[@]}" ;;
      zypper) $SUDO zypper --non-interactive install "${NEED[@]}" ;;
      brew)   brew install "${NEED[@]}" ;;
    esac
    ok "native packages installed"
  else
    warn "skipped — Box2D (LunarLander) and/or the MuJoCo viewer may not work"
  fi
fi

# ============================================================================
step "3/5  Python interpreter"
# ============================================================================
if [[ -z "$PYTHON_BIN" ]]; then
  for cand in python3.12 python3.11 python3.10 python3; do
    if command -v "$cand" >/dev/null; then PYTHON_BIN="$cand"; break; fi
  done
fi
command -v "$PYTHON_BIN" >/dev/null || die "no suitable python found (need >= 3.10)"

PYVER="$("$PYTHON_BIN" -c 'import sys; print("%d.%d"%sys.version_info[:2])')"
"$PYTHON_BIN" -c 'import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)' \
  || die "$PYTHON_BIN is $PYVER — mujoco, gymnasium and SB3 all need >= 3.10"
ok "$PYTHON_BIN ($PYVER)"

# ============================================================================
step "4/5  Virtualenv + Python packages"
# ============================================================================
if [[ $CHECK_ONLY -eq 1 ]]; then
  [[ -d "$VENV_DIR" ]] && ok ".venv exists" || warn ".venv missing (would create)"
else
  if [[ ! -d "$VENV_DIR" ]]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR" || die "venv creation failed (install python3-venv)"
    ok "created $VENV_DIR"
  else
    ok "reusing $VENV_DIR"
  fi

  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  python -m pip install --upgrade pip setuptools wheel --quiet
  ok "pip $(python -m pip --version | awk '{print $2}')"

  # PyTorch first, from the flavour-specific index. Doing it in a separate
  # step stops pip from pulling the wrong ~2.5 GB wheel as a transitive dep
  # of something else in requirements.txt.
  case "$TORCH_FLAVOUR" in
    cuda)
      info "installing CUDA PyTorch from ${TORCH_CUDA_INDEX} (large download)"
      python -m pip install "torch>=2.4,<3" --index-url "$TORCH_CUDA_INDEX"
      ;;
    xpu)
      info "installing Intel XPU PyTorch from ${TORCH_XPU_INDEX} (large download)"
      python -m pip install "torch>=2.4,<3" --index-url "$TORCH_XPU_INDEX"
      ;;
    rocm)
      info "installing ROCm PyTorch from ${TORCH_ROCM_INDEX} (large download)"
      python -m pip install "torch>=2.4,<3" --index-url "$TORCH_ROCM_INDEX"
      ;;
    none)
      info "skipping torch install (--gpu none)"
      ;;
    *)
      info "installing CPU-only PyTorch from ${TORCH_CPU_INDEX}"
      python -m pip install "torch>=2.4,<3" --index-url "$TORCH_CPU_INDEX"
      ;;
  esac
  [[ "$TORCH_FLAVOUR" != "none" ]] && ok "torch installed (${TORCH_FLAVOUR})"

  info "installing the rest of requirements.txt"
  python -m pip install -r "${REPO_ROOT}/requirements.txt"
  ok "python packages installed"
fi

# ============================================================================
step "5/5  Verification"
# ============================================================================
VERIFY_PY="$PYTHON_BIN"
[[ -x "$VENV_DIR/bin/python" ]] && VERIFY_PY="$VENV_DIR/bin/python"
set +e
"$VERIFY_PY" "${REPO_ROOT}/scripts/check_env.py"
RC=$?
set -e

cat <<EOF

${B}Next steps${N}
  ${D}1.${N} Activate the environment:      ${B}source .venv/bin/activate${N}
  ${D}2.${N} Re-verify any time:            ${B}python scripts/check_env.py${N}
  ${D}3.${N} Run something that needs no policy weights:
         ${B}python stage2-go2-mujoco-inference/01_hello_go2.py${N}        ${D}(viewer)${N}
         ${B}python stage2-go2-mujoco-inference/experiments/make_figures.py${N} ${D}(figures)${N}
  ${D}4.${N} Full Stage 2 inference needs the pretrained policy — see
         ${B}docs/DEPENDENCIES.md${N} (section "The one dependency we cannot install")

EOF
exit $RC
