#!/bin/bash
declare -A dictionary
dictionary["alcock"]="1PaZ7hVp_13WyilLZlWlWNOZhCEbsKopA"
dictionary["macho"]="1MInBPl0mt4Eerq5zxQsPMgJU-CbwLc8u"
dictionary["ogle"]="1B8fgyPrNNR5FQdGwPAIWmMk87aL1RZRD"
dictionary["atlas"]="125euwxpF0WY_oWq3jVtq9c5pNyUISAOO"

dictionary["alcock-record"]="1N4-n3OUAA0J2jFF6fO2MmbykzcxzLEuX"
dictionary["atlas-record"]="1OnSF3EULdJpFY4MwRZLgmXa6DauziR8W"

FILEID=${dictionary[$1]}
echo $FILEID

SUB="record"
if [[ "$1" == *"$SUB"* ]];
then
    NAME=${1%-*}
    mkdir -p records/
    DIR=./records/
    OUTFILE=./records/$NAME.zip
    echo $OUTFILE
else
    mkdir -p raw_data/$1
    DIR=./raw_data/$1
    OUTFILE=./raw_data/$1/$1.zip
fi

gdown https://drive.google.com/uc?id=$FILEID -O $OUTFILE -c

unzip $OUTFILE -d $DIR
rm -rf $OUTFILE