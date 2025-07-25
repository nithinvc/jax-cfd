#!/bin/bash

END=10
jid=`sbatch ./generate_multiparameter_nersc.sh | sed 's/Submitted batch job //'`
for i in {1..$END}; do jid=`sbatch --dependency=afterany:$jid ./generate_multiparameter_nersc.sh | sed 's/Submitted batch job //'`; done
