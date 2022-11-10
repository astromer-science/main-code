# ASTROMER

<p align="center">
  <img src="https://github.com/cridonoso/astromer/blob/main/presentation/figures/astromer_logo.png?raw=true" width="1300" title="hover text">
</p>

Welcome to the main repository of the ASTROMER project. This is the **stable
version 0** where you can find all the resources associated to the article: 
[ASTROMER: A transformer-based embedding for the representation of light curves](https://arxiv.org/abs/2205.01677).

ASTROMER is a transformer-based model that learns **light curves representations** 
using millions of light curves. 

Representation are then used for extracting 
useful **embeddings** that we can use for training another downstream tasks.






## Features

- Pre-trained weights from MACHO R-band light curves
- Visualization jupyter notebooks (`/presentation/notebooks/*`)
- Scripts for training, finetuning and classify using ASTROMER (`/presentation/scripts/*`)
- Model implementation (`/core/astromer.py` and related)
- Data preprocessing, saving and reading [tf.Records](https://www.tensorflow.org/tutorials/load_data/tfrecord) (`/core/data.py`)
- Dockerfile and scripts for building (`build_container.sh`) and run (`run_container.sh`) the ASTROMER container

## Get started

We recomend to use [Docker](https://docs.docker.com/get-docker/) since it provides a **kernel-isolated** 
and **identical environment** to the one used by the authors

The `Dockerfile` contains all the configuration for running ASTROMER model. No need to touch it,
`build_container.sh` and `run_container.sh` make the work for you :slightly_smiling_face:	

The first step is to build the container,
```bash
  bash build_container.sh
```
It creates a "virtual machine", named `astromer`, containing all the dependencies such as python, tensorflow, among others. 

The next and final step is running the ASTROMER container,
```
  bash run_container.sh
```
The above script looks for the container named `astromer` and run it on top of [your kernel](https://www.techtarget.com/searchdatacenter/definition/kernel#:~:text=The%20kernel%20is%20the%20essential,systems%2C%20device%20control%20and%20networking.).
Automatically, the script recognizes if there are GPUs, making them visible inside the container.

By default the `run_container.sh` script opens the ports `8888` and `6006` 
for **jupyter notebook** and [**tensorboard**](https://github.com/cridonoso/tensorboard_tutorials), resepectively.
To run them, use the usal commands but adding the following lines:

For Jupyter Notebook 
```
jupyter notebook --ip 0.0.0.0
```
(Optionally) You can add the `--no-browser` tag in order to avoid warnings.

For Tensorboard
```
tensorboard --logdir <my-logs-folder> --host 0.0.0.0
```

Finally, **if you do not want to use Docker** the `requirements.txt` file contains 
all the packages needed to run ASTROMER.
Use `pip install -r requirements.txt` on your local python to install them.
## Usage/Examples

We recomend to save data in tf.Records format.
For creating records jump to the [create records tutorial](https://github.com/cridonoso/astromer/blob/main/presentation/notebooks/create_records.ipynb)

Otherwise, if you have numpy-based light curves, use [`load_numpy()`](https://github.com/cridonoso/astromer/blob/main/core/data.py) function.

For pre-training run [`train.py`](https://github.com/cridonoso/astromer/blob/main/presentation/scripts/train.py)
```
python -m presentation.scripts.train --data ./data/records/macho
```
To see the `--tag` options use `--help`. For example, 
```
python -m presentation.scripts.finetuning --help
```
The [pre-training](https://github.com/cridonoso/astromer/blob/main/presentation/scripts/train.py), 
[finetuning](https://github.com/cridonoso/astromer/blob/main/presentation/scripts/finetuning.py), 
and [classification](https://github.com/cridonoso/astromer/blob/main/presentation/scripts/classify.py) 
scripts work in the same way.

## Contributing

Contributions are always welcome!

Issues and featuring can be directly published in this repository
via [Pull Request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests). 
Similarly, New pre-trained weights must be uploaded to the [weights repo](https://github.com/astromer-science/weights) using the same mechanism.

Look at [this tutorial](https://cridonoso.github.io/articles/github.html) for more information about pull requests
