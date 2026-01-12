# Creating Records

This tutorial explains how to generate the records needed to train the models, from downloading the raw data to the final verification.

---

## 1. Download the Raw Data

First, you need to download the raw data for the desired survey. We use the `get_data.sh` script for this, which automates the process of downloading and decompressing the data from Google Drive.

Here is a preview of the script's content:

```bash
#!/bin/bash
declare -A dictionary
dictionary["alcock"]="1PaZ7hVp_13WyilLZlWlWNOZhCEbsKopA"
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
```

To use it, simply run the following command from the root of the repository, replacing `<dataset_name>` with the name of the data you want to download (e.g., `alcock`, `macho`, `ogle`).

```bash
bash data/get_data.sh <dataset_name>
```

This command will automatically create a `raw_data/<dataset_name>` directory and download the necessary files into it.

---

## 2. Configure your Dataset

To process the raw data, you need a `config.toml` file. This file specifies the structure of your data. You can copy the example file located in `data/config.toml` and modify it for your specific needs. You can place this configuration file wherever you prefer.

Here is an explanation of each field in the `config.toml` file:

```toml
[id_column]
value = "newID"
dtype = "integer"
```
* **`[id_column]`**: Defines the unique identifier for each time series.
    * `value`: The name of the column in your metadata that contains the ID.
    * `dtype`: The data type of the ID column.

```toml
[target]
path = "./data/records/macho/" # can be overwritten
dtype = "string"
```
* **`[target]`**: Specifies where the generated records will be saved.
    * `path`: The output directory for the records.
    * `dtype`: The data type of the target variable (not used in the script, but good practice to define).

```toml
[context_features]
path = "./data/raw_data/macho/cleaned_metadata.parquet"
test_path = "./data/raw_data/macho/test_metadata.parquet"
value = [ "ID", "Class", "Band", "Label", "shard",]
dtypes = [ "string", "string", "string", "integer", "integer",]
```
* **`[context_features]`**: Describes the contextual or static data (metadata).
    * `path`: Path to the metadata file (usually a `.parquet` file).
    * `test_path`: (Optional) Path to a separate metadata file for the test set.
    * `value`: A list of the column names to be used as context features.
    * `dtypes`: A list of the corresponding data types for the context features.

```toml
[sequential_features]
path = "./data/raw_data/macho/light_curves"
value = [ "mjd", "mag", "errmag"]
dtypes = [ "float", "float", "float"]
```
* **`[sequential_features]`**: Describes the time-series data (light curves).
    * `path`: Path to the directory containing the light curve files.
    * `value`: A list of the column names within each light curve file.
    * `dtypes`: A list of the corresponding data types for the sequential features.

Once you have configured the file with the correct paths and column names for your data, save it.

---

## 3. Generate the Records

With the `config.toml` file ready, you can now generate the records. From the root of the repository, execute the following script:

```bash
python -m presentation.scripts.create_records --config /path/to/your/config.toml
```

The `create_records.py` script also has other optional parameters you can use:

* `--folds`: Number of folds to create for cross-validation. Default is `1`.
* `--val-frac`: Fraction of the data to be used for the validation set. Default is `0.2`.
* `--test-frac`: Fraction of the data to be used for the test set. Default is `0.2`.
* `--njobs`: Number of CPU cores to use for parallel processing. Default is `4`.
* `--elements-per-shard`: Number of light curves to store in each output file (shard). Default is `200000`.

After the process is complete, the records will have been created in the output directory specified in your `config.toml`.

---

## 4. Verify the Records

To ensure that the records were created correctly, you can run the corresponding tests using `pytest`. Execute the test designed to verify the integrity of the records.

```bash
pytest -s testing/unit_tests/test_records.py --config-path ./data/raw_data/alcock/config.toml
```
Where `--config-path` is the config path we used before for creating records. 

If all tests pass, your records are ready to be used! ✅