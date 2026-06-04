#!/bin/bash
# HDF5 over XRootD smoke test
export LD_PRELOAD=/usr/lib64/libXrdPosixPreload.so
export XROOTD_VMP="fndcadoor.fnal.gov:1094:/pnfs=/pnfs/fnal.gov/usr"

python -c "
import h5py
p = '/pnfs/sbnd/persistent/users/brindenc/genie/crpa/gevgens/numu/prism_processed/100k/crpa.df'
with h5py.File(p, 'r') as f:
    print(list(f.keys())[:5])
"
