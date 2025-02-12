<p align="center">
    <img src="./presentation/figures/branding/banner.png"> 
</p>

![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)

This is the latest source code of <b>Astromer</b> used by authors to run experiments and analysis. Use this code if you want to adapt or extend functionalities of <b>Astromer</b>. Our Python package contains a snapshot of this repository in its stable version. 

## About ASTROMER

ASTROMER is a deep learning model that can encode single-band light curves using internal representations. The encoding process consists of creating embeddings of light curves, which are features that summarize the variability of the brightness over time.   
<p align="center">
<img src="./presentation/figures/branding/astromer-arch-trans.png"> 
</p>
Training ASTROMER from scratch can be expensive. However, we provide pre-trained weights that can be used to load representations already adjusted on millions of samples. Users can then easily fine-tune the model on new data. Fine-tuning involves training the pre-trained model on new datasets, which are typically smaller than the pre-training dataset.

## Weights

| Version Tag | Pretraining data | Description | Test RMSE/R-square | Link |
| --- | --- | --- | --- | --- |
| v0 | MACHO | Paper's model | 0.147/0.80 | [Download Weights](https://github.com/astromer-science/weights/raw/main/macho_a0.zip)
| v1*  | MACHO | Mask token and residual connections. | 0.113/0.73 | [Download Weights](https://github.com/astromer-science/weights/raw/main/macho_a1.zip)

\* best performance up to date
## Directory tree
```
📦astromer
 ┣ 📂data
 ┃ ┣ 📜 get_data.sh: download data (raw and preprocessed) from gdrive
 ┃ ┣ 📜 config.toml: config template for converting parquet to records
 ┃ ┗ 📜 README.md: instructions how to use get_data.sh 
 ┣ 📂 presentation: everything that depends on the model code (i.e., pipelines, figures, visualization notebooks...)
 ┃ ┣ 📂 results: experiments folder
 ┃ ┣ 📂 figures
 ┃ ┣ 📂 notebooks: util notebooks to visualize data, model features and results
 ┃ ┣ 📂 pipelines:
 ┃ ┃ ┗ 📂 pipeline_0
 ┃ ┃ ┃ ┗ 📜 finetune.py: Script for running finetunining step on a pretrained model
 ┃ ┃ ┃ ┗ 📜 classify.py: Script for running classification on a pretrained/finetuned model
 ┃ ┃ ┃ ┗ 📜 utils.py: utils functions for pipeline_0
 ┃ ┃ ┃ ┗ 📜 run-gpu-{ID}.bash: bash script to run the whole pipeline on several pre-trained models
 ┃ ┃ ┗ 📂 steps: functions that are invariant to the pipeline and will be always used 
 ┃ ┃ ┃ ┗ 📜 load_data.py: functions to easily load records
 ┃ ┃ ┃ ┗ 📜 metrics.py: functions to get general metrics and tensorboard logs 
 ┃ ┃ ┃ ┗ 📜 model_design.py: function to easily load Astromer
 ┃ ┃ ┃ ┗ 📜 utils.py: function for custom training [DEPRECATED]
 ┃ ┃ ┗ 📜 utils.py: useful and general functions to create pipelines
 ┃ ┣ 📂 scripts:
 ┃ ┃ ┗ 📜 create_records.py: Script for creating records. It assumes data is in parquet format using the standard structure.
 ┃ ┃ ┗ 📜 create_sset.py: Script for creating pretraining subsets with different number of samples  
 ┃ ┃ ┗ 📜 pretrain.py: Script for pretrain a model from scratch
 ┃ ┃ ┗ 📜 test_model.py: Evaluate a pre-trained model on a testing set (tf.record)
 ┃ ┃ ┗ 📜 to_parquet.py: Transform old-version raw data to new structure based on parquet files
 ┣ 📂 src: Model source code
 ┃ ┗ 📂 data: functions related to data manipulation
 ┃ ┃ ┣ 📜 loaders.py: main script containing functions to load and format data
 ┃ ┃ ┣ 📜 masking.py: masking functions inspired on BERT training strategy
 ┃ ┃ ┣ 📜 preprocessing.py: general functions to standardize, cut windows, among others.
 ┃ ┃ ┗ 📜 record.py: functions to create and load tensorflow record files
 ┃ ┃ ┗ 📜 zero.py: old functions to load tf.records. [not being used in this implementation]
 ┃ ┗ 📂 layers: Custom layers used to build ASTROMER model
 ┃ ┃ ┣ 📜 attblock.py: Attention block definition. Each block contain self-attention heads, normalization and transformation layers.
 ┃ ┃ ┣ 📜 attention.py: Multi-head self-attention layers.
 ┃ ┃ ┣ 📜 custom_rnn.py: Normalized LSTM [not being used in this implementation].
 ┃ ┃ ┣ 📜 encoders.py: Astromer encoder definition. It also contains encoder alternatives that inherit from the parent class.
 ┃ ┃ ┣ 📜 input.py: Input transformation layers
 ┃ ┃ ┣ 📜 output.py: Layers that take the embeddings and project them to desired outputs (regression/classification)
 ┃ ┃ ┗ 📜 positional.py: Positional encoder class
 ┃ ┗ 📂 losses
 ┃ ┃ ┣ 📜 bce.py: Masked binary cross-entropy (used with NSP)
 ┃ ┃ ┗ 📜 rmse.py: Masked root mean square error
 ┃ ┗ 📂 metrics
 ┃ ┃ ┣ 📜 acc.py: masked accuracy
 ┃ ┃ ┗ 📜 r2.py: masked r-square
 ┃ ┗ 📂 models: ASTROMER model architectures
 ┃ ┃ ┣ 📜 astromer_0.py: Astromer v0 (Donoso et.al. 2023)
 ┃ ┃ ┣ 📜 astromer_1.py: Astromer v1 (Donoso et.al. 2024 in PROGRESS)
 ┃ ┗ 📂 training
 ┃ ┃ ┗ 📜 scheduler.py: Custom scheduler presented in https://arxiv.org/abs/1706.03762
 ┃ ┣ 📜 __init__.py
 ┃ ┗ 📜 utils.py: universal functions to use on different ASTROMER modules
 ┣ 📜 .gitignore: files that should not be considered during a GitHub push
 ┣ 📜 requirements.txt: python dependencies
 ┗ 📜 README.md: what you are currently reading
 ```
## Get started

Use conda to create a virtual environment, for example:
```
conda create -n astromer python==3.9
```
Then install all the python packages use `requirements.txt`
```
pip install -r requirements.txt
```


## Contributing

Contributions are always welcome!

Issues and featuring can be directly published in this repository
via [Pull Request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests). 
Similarly, New pre-trained weights must be uploaded to the [weights repo](https://github.com/astromer-science/weights) using the same mechanism.
