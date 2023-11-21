#!/bin/bash

# We assume running this from the script directory
job_directory=$PWD/

#!/bin/bash
for i in {1..30}
do
   # # sleep 1s # Waits x seconds.
   echo "Run Script $i Times"
   sed -i "s/^INPUTFILE=.*py$/INPUTFILE=\$(pwd)\/extra_trees_RFE_y_permuted_thermal_$i.py/g" launcher.sh
   sbatch --cpus-per-task=16 --time=01:15:00 launcher.sh
done
