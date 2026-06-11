#!/usr/bin/env bash
# Pass through extra args, e.g.  . runit.sh --dry-run  or  bash runit.sh --dry-run --only jobname
#source /exp/sbnd/app/users/brindenc/develop/cafpyana/setup.sh
# Test
python /exp/sbnd/app/users/brindenc/develop/cafpyana/analysis_village/numuincl/runit.py -y /exp/sbnd/app/users/brindenc/develop/cafpyana/analysis_village/numuincl/yamls/mcbnb.yaml
#CV
#python /exp/sbnd/app/users/brindenc/develop/cafpyana/analysis_village/numuincl/runit.py -y /exp/sbnd/app/users/brindenc/develop/cafpyana/analysis_village/numuincl/yamls/mcbnb_nosyst.yaml
# python /exp/sbnd/app/users/brindenc/develop/cafpyana/analysis_village/numuincl/runit.py -y /exp/sbnd/app/users/brindenc/develop/cafpyana/analysis_village/numuincl/yamls/mcbnb_fullsyst.yaml
# python /exp/sbnd/app/users/brindenc/develop/cafpyana/analysis_village/numuincl/runit.py -y /exp/sbnd/app/users/brindenc/develop/cafpyana/analysis_village/numuincl/yamls/mcbnb_slimsyst.yaml
# python /exp/sbnd/app/users/brindenc/develop/cafpyana/analysis_village/numuincl/runit.py -y /exp/sbnd/app/users/brindenc/develop/cafpyana/analysis_village/numuincl/yamls/mcoffbeam.yaml
# python /exp/sbnd/app/users/brindenc/develop/cafpyana/analysis_village/numuincl/runit.py -y /exp/sbnd/app/users/brindenc/develop/cafpyana/analysis_village/numuincl/yamls/mclowe.yaml
# # Data
# python /exp/sbnd/app/users/brindenc/develop/cafpyana/analysis_village/numuincl/runit.py -y /exp/sbnd/app/users/brindenc/develop/cafpyana/analysis_village/numuincl/yamls/dataoffbeam.yaml
#python /exp/sbnd/app/users/brindenc/develop/cafpyana/analysis_village/numuincl/runit.py -y /exp/sbnd/app/users/brindenc/develop/cafpyana/analysis_village/numuincl/yamls/data.yaml
# # Det var
# python /exp/sbnd/app/users/brindenc/develop/cafpyana/analysis_village/numuincl/runit.py -y /exp/sbnd/app/users/brindenc/develop/cafpyana/analysis_village/numuincl/yamls/det_var/nominal.yaml
# python /exp/sbnd/app/users/brindenc/develop/cafpyana/analysis_village/numuincl/runit.py -y /exp/sbnd/app/users/brindenc/develop/cafpyana/analysis_village/numuincl/yamls/det_var/pmtgain.yaml
# python /exp/sbnd/app/users/brindenc/develop/cafpyana/analysis_village/numuincl/runit.py -y /exp/sbnd/app/users/brindenc/develop/cafpyana/analysis_village/numuincl/yamls/det_var/pmtqe.yaml
# python /exp/sbnd/app/users/brindenc/develop/cafpyana/analysis_village/numuincl/runit.py -y /exp/sbnd/app/users/brindenc/develop/cafpyana/analysis_village/numuincl/yamls/det_var/pmtspe.yaml
# python /exp/sbnd/app/users/brindenc/develop/cafpyana/analysis_village/numuincl/runit.py -y /exp/sbnd/app/users/brindenc/develop/cafpyana/analysis_village/numuincl/yamls/det_var/nosce.yaml
# python /exp/sbnd/app/users/brindenc/develop/cafpyana/analysis_village/numuincl/runit.py -y /exp/sbnd/app/users/brindenc/develop/cafpyana/analysis_village/numuincl/yamls/det_var/twicesce.yaml
# python /exp/sbnd/app/users/brindenc/develop/cafpyana/analysis_village/numuincl/runit.py -y /exp/sbnd/app/users/brindenc/develop/cafpyana/analysis_village/numuincl/yamls/det_var/wiremodxtheta.yaml
# python /exp/sbnd/app/users/brindenc/develop/cafpyana/analysis_village/numuincl/runit.py -y /exp/sbnd/app/users/brindenc/develop/cafpyana/analysis_village/numuincl/yamls/det_var/wiremodyz.yaml