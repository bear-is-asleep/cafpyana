#!/bin/bash
# Minimal cafpyana env for Jupyter (jupyterlab in envs/pip_requirements.txt)

export machine=${HOSTNAME}
export CAFPYANA_DIR=$(pwd)

SPACK_ROOT="/cvmfs/larsoft.opensciencegrid.org/spack-fnal-v1.0.0/setup-env.sh"
source "${SPACK_ROOT}"

if [[ $machine == *jupyter* ]]; then
  export CAFPYANA_GRID_OUT_DIR="/pnfs/sbnd/scratch/users/$USER/cafpyana_out"
  export CAFPYANA_TMP_SCRATCH="/scratch/7DayLifetime/$USER/tmp_failover"
  mkdir -p "$CAFPYANA_GRID_OUT_DIR" "$CAFPYANA_TMP_SCRATCH"
fi

spack load root arch=linux-almalinux9-x86_64_v2

ENV_DIR=${CAFPYANA_DIR}/envs
LOGDIR=${ENV_DIR}/logs
mkdir -p "${LOGDIR}"

VENV_NAME=venv_py310_cafpyana
VENV_DIR="${ENV_DIR}/${VENV_NAME}"

[ -d "${VENV_DIR}" ] || python -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

PYTHONPATH=$(python -c 'import os; print(":".join(p for p in os.environ.get("PYTHONPATH", "").split(":") if p and not ("/xrootd-" in p and "/site-packages" in p)))')
export PYTHONPATH

if pip freeze -r "${ENV_DIR}/pip_requirements.txt" 2>&1 | grep -q "not installed"; then
  PIPLOG=${LOGDIR}/init_pip.log
  echo "$(date)" >> "${PIPLOG}"
  pip install --upgrade pip wheel setuptools | tee -a "${PIPLOG}"
  pip install -r "${ENV_DIR}/pip_requirements.txt" | tee -a "${PIPLOG}"
fi

if [[ $machine == *jupyter* ]]; then
  export C_INCLUDE_PATH="${ENV_DIR}/local/include:${C_INCLUDE_PATH}"
  export CPLUS_INCLUDE_PATH="${ENV_DIR}/local/include:${CPLUS_INCLUDE_PATH}"
  export LD_LIBRARY_PATH="${ENV_DIR}/local/lib:${LD_LIBRARY_PATH}"
  export PKG_CONFIG_PATH="${ENV_DIR}/local/lib/pkgconfig:${PKG_CONFIG_PATH}"

  if [ ! -f "${ENV_DIR}/local/lib/libuuid.so" ]; then
    cd "${ENV_DIR}" || exit 1
    UUIDLOG=${LOGDIR}/init_uuid.log
    echo "$(date)" >> "${UUIDLOG}"
    wget -q https://www.kernel.org/pub/linux/utils/util-linux/v2.39/util-linux-2.39.3.tar.xz
    tar xf util-linux-2.39.3.tar.xz && rm util-linux-2.39.3.tar.xz
    cd util-linux-2.39.3
    ./configure --prefix="${ENV_DIR}/local" --disable-all-programs --enable-libuuid | tee -a "${UUIDLOG}"
    make -j"$(nproc)" install | tee -a "${UUIDLOG}"
    cd "${CAFPYANA_DIR}" || exit 1
  fi
fi

python -c "import XRootD" > /dev/null 2>&1 || {
  echo "Could not import XRootD! Attempting to install from source..."
  cd "${ENV_DIR}" || exit 1
  XROOTLOG=${LOGDIR}/init_xroot.log
  echo "$(date)" >> "${XROOTLOG}"

  OLDPATH=$PATH
  PATH=$PATH:${ENV_DIR}
  ln -sf /cvmfs/larsoft.opensciencegrid.org/products/cmake/v3_22_2/Linux64bit+3.10-2.17/bin/cmake "${ENV_DIR}/cmake3"
  echo "Using cmake at $(which cmake3)" | tee -a "${XROOTLOG}"
  wget -q https://files.pythonhosted.org/packages/fd/4f/419b8caec575ab4133f41c37c39bd742251a4dc6a208a97a3fd772031fe7/xrootd-5.6.9.tar.gz
  tar -zxf xrootd-5.6.9.tar.gz && rm xrootd-5.6.9.tar.gz
  cd xrootd-5.6.9
  sed -i 's/SSL_CTX_flush_sessions/SSL_CTX_flush_sessions_ex/g' src/XrdTls/XrdTlsContext.cc
  python setup.py install 2>&1 | tee -a "${XROOTLOG}"
  cd "${CAFPYANA_DIR}" || exit 1
  PATH=$OLDPATH
}

PYXROOTD_DIR=$(python -c "import glob, sys; print(next(iter(glob.glob(f'{sys.prefix}/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/xrootd-*/pyxrootd')), ''))")
if [ -n "${PYXROOTD_DIR}" ]; then
  export LD_LIBRARY_PATH="${PYXROOTD_DIR}:${LD_LIBRARY_PATH}"
fi

export PYTHONPATH="${PYTHONPATH}:${CAFPYANA_DIR}"
export CAFPYANA_WD=${CAFPYANA_DIR}

#htgettoken -a htvaultprod.fnal.gov -i sbnd