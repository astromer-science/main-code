#!/bin/bash
declare -A dictionary
dictionary["catalina"]="1rHJpj4eUUOuuGJlEFSPmBEirF5Ww26Nk"
dictionary["alcock"]="18a4DGPlyJ21DI9HrKS-jJtaA2iIqFDQO"
dictionary["macho"]="1MInBPl0mt4Eerq5zxQsPMgJU-CbwLc8u"
dictionary["ogle"]="1B8fgyPrNNR5FQdGwPAIWmMk87aL1RZRD"
dictionary["atlas"]="125euwxpF0WY_oWq3jVtq9c5pNyUISAOO"

dictionary["alcock-record"]="1bEETbIgsVjhpkfR0LdYxnYQ8eeaq4wol"
dictionary["macho-record"]="1QLXAsTkaryYUqhjKAM0wh6XKh3tG1M0k"
dictionary["ogle-record"]="1Ei5PZ13LjJ44tA2iBOkDHPueGbLlTl2C"
dictionary["atlas-record"]="1e6dtsaidOBnbVo5gP8IkZXD_8ooiLL6d"

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
