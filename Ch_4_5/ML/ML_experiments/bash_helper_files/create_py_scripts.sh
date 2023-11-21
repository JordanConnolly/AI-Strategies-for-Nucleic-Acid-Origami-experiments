#!/bin/bash

# We assume running this from the script directory
job_directory=$PWD/

#!/bin/bash
for i in {1..30}
do
   # create a copy of the original scripts
   
   # cp extra_trees_RFE_no_revnano.py "extra_trees_RFE_no_revnano_$i.py"
   cp extra_trees_RFE.py "extra_trees_RFE_$i.py"

   echo "Created Custom Params Script $i Times"
done
