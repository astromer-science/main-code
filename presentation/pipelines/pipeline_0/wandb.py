

#!/usr/bin/python
import subprocess
import os
import sys
import time

from tqdm import tqdm
from functools import partial


gpu = sys.argv[1]

for i in range(10):
    
    command1 = 'python -m presentation.pipelines.pipeline_0.hp \
                                        --data ./data/records/macho/100000/fold_0/ \
                                        --bs 2500 \
                                        --gpu {}'.format(gpu)

    
    try:
        subprocess.call(command1, shell=True)
    except Exception as e:
        print(e)





