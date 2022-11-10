#!/usr/bin/python
import subprocess
import sys
import time
import json
import os, sys

gpu = sys.argv[1]
ds_name = sys.argv[2]
batch_size = 512

datasets = ['{}_20'.format(ds_name),
            '{}_50'.format(ds_name),
            '{}_100'.format(ds_name),
            '{}_500'.format(ds_name),
            ]
for astromer_dim in [256, 128, 64]:
    for science_case in ['a', 'b', 'c']:

        if science_case == 'a':
            train_astromer = False
            sc_ft = 'ab'
        if science_case == 'b':
            train_astromer = True
            sc_ft = 'ab'
        if science_case == 'c':
            train_astromer = True
            sc_ft = 'c'

        for dataset in datasets:
            for fold_n in range(3):
                for mode in ['lstm_att', 'mlp_att', 'lstm']:
                    print('sc:{} - {} on mode {}'.format(science_case, dataset, mode))

                    if science_case == 'c':
                        astroweights = './runs/astromer_{}/{}/{}/fold_{}/{}'.format(astromer_dim,
                                                                                    sc_ft,
                                                                                    ds_name,
                                                                                    fold_n,
                                                                                    ds_name)
                    else:
                        astroweights = './runs/astromer_{}/{}/{}/fold_{}/{}'.format(astromer_dim,
                                                                sc_ft,
                                                                ds_name,
                                                                fold_n,
                                                                dataset)

                    project_path = './runs/astromer_{}/classifiers/{}/{}/fold_{}/{}'.format(astromer_dim,
                                                                                            science_case,
                                                                                            ds_name,
                                                                                            fold_n,
                                                                                            dataset)

                    command1 = 'python -m presentation.scripts.classify \
                                    --data ./data/records/{}/fold_{}/{} \
                                    --p {} \
                                    --w {} \
                                    --batch-size {} \
                                    --mode {} \
                                    --gpu {}'.format(ds_name, fold_n, dataset,
                                                     project_path,
                                                     astroweights,
                                                     batch_size,
                                                     mode,
                                                     gpu)
                    if train_astromer:
                        command1 += ' --finetune'

                    try:
                        subprocess.call(command1, shell=True)
                    except Exception as e:
                        print(e)
