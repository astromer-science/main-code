#!/bin/bash

echo Getting ids...
python -m presentation.pipelines.machofields.get_ids

while IFS=',' read -r col1 field_n idcode
do  
    if [ -d "./data/shared/records/bigmacho/$field_n/" ]; then
        echo "The folder exists."
    else
        if [ "$col1" != "" ]; then
            echo Processing $field_n
            python -m presentation.pipelines.machofields.download --id $idcode \
                                                                  --field $field_n \
                                                                  --target ./data/shared/records/bigmacho
            echo Cleaning Light Curves based on Variability Criteria...
            python -m presentation.scripts.clean_pipeline --data ./data/temp/
            
            echo Transforming to records...
            python -m presentation.pipelines.machofields.to_record --config ./data/shared/records/bigmacho/$field_n/config.toml
    
            echo 'Deleting raw data'
            rm -rf ./data/temp/light_curves
            rm -rf ./data/temp/metadata.parquet                                               
        fi
    fi
        
done < ./data/temp/ids.csv