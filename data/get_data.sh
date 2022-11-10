#!/bin/bash
declare -A dictionary
dictionary["alcock"]="18a4DGPlyJ21DI9HrKS-jJtaA2iIqFDQO"
dictionary["macho"]="1ReqDHn9jKPIS_g8Xm0ThYUDM-XaLcUP3"
dictionary["ogle"]="1L1oiq9pRRGpOVm13b2dyCQhMiVFn3o7x"
dictionary["atlas"]="1pMzeL9BAwMXqra9iFUHLpGyclJzz-VM9"

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

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$FILEID -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$FILEID" -O $OUTFILE && rm -rf /tmp/cookies.txt


unzip $OUTFILE -d $DIR
rm -rf $OUTFILE
