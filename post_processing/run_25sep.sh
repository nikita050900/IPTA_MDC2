#!/bin/bash
# run_25sep.sh : runs both 25 Sep analyses on an Anvil login node
cd /anvil/projects/x-phy260442/projects/IPTA_MDC2/post_processing
module load anaconda/2024.02-py311
source activate qcw_eccentric
nohup python -u h0_boot_CD_25sep.py    > h0_boot_CD_25sep.out    2>&1 &
nohup python -u dl_param_check_25sep.py > dl_param_check_25sep.out 2>&1 &
echo "started on $(hostname)"; jobs -l
