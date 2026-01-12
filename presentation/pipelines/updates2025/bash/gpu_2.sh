#!/bin/bash
                 
folds=('2')
datasets=('atlas')
spcs=('20' '100' '500')
model_path='./presentation/results/diagstromer/2024-12-02_14-13-12/finetuning'
dataroot='/home/shared/astromer/records'
# 'base_avgpool' 'base_gru' 'max' 'avg' 'skip' 'att_avg' 'att_cls' raw_gru
clf_models=('raw_gru' 'base_avgpool' 'base_gru' 'max' 'avg' 'skip' 'att_avg' 'att_cls')
for fold_N in ${folds[@]}; do
    for dp in ${datasets[@]}; do
        for spc in ${spcs[@]}; do
            for clfmodel in ${clf_models[@]}; do
                echo [INFO] Starting CLF $fold_N $spc $clfmodel
                python -m presentation.pipelines.referee.train \
                --pt-path $model_path/$dp/fold_$fold_N/$dp\_$spc/ \
                --data $dataroot/$dp/fold_$fold_N/$dp\_$spc \
                --gpu 2 \
                --bs 3000 \
                --lr 0.001 \
                --exp-name clf_$dp\_$fold_N\_$spc \
                --clf-arch $clfmodel
                done
            done
    done
done
