#!/usr/bin/python
import subprocess
import sys
import time
import json
import os, sys

gpu = sys.argv[1]
ds_name = sys.argv[2]
science_case = sys.argv[3]

astromer_dim = sys.argv[4]
astroweights = './weights/macho_{}'.format(astromer_dim)
batch_size = 2500

if science_case == 'c':
    datasets = [ds_name]
else:
    science_case = 'ab'
    datasets = ['{}_20'.format(ds_name),
                '{}_50'.format(ds_name),
                '{}_100'.format(ds_name),
                '{}_500'.format(ds_name)]

conf_file = os.path.join(astroweights, 'conf.json')
with open(conf_file, 'r') as handle:
    conf = json.load(handle)

for dataset in datasets:
    print(dataset)
    for fold_n in range(3):
        start = time.time()
        project_path = './runs/astromer_{}/{}/{}/fold_{}/{}'.format(astromer_dim,
                                                                    science_case,
                                                                    ds_name,
                                                                    fold_n,
                                                                    dataset)

        command1 = 'python -m presentation.scripts.finetuning \
                   --data ./data/records/{}/fold_{}/{} \
                   --w {} \
                   --p {} \
                   --gpu {}\
                   --batch-size {}'.format(ds_name, fold_n, dataset,
                                    astroweights,
                                    project_path,
                                    gpu,
                                    batch_size)
        try:
            subprocess.call(command1, shell=True)
        except Exception as e:
            print(e)

        end = time. time()
        print('{} takes {:.2f} sec'.format(dataset, (end - start)))
