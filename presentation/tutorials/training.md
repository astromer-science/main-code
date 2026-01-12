# Pre-training the AstroMer Model

Pre-training is the first and most crucial step in creating a powerful contextual representation of time series data. This process teaches the model the underlying patterns and dynamics of sequential data before fine-tuning it for a specific downstream task.

---
## Single-GPU Pre-training

To start the pre-training process on a single GPU, you can use the `pretrain.py` script. The only required argument is the path to the folder containing your previously generated records.

### Command
```bash
python -m presentation.scripts.pretrain --data ./data/records/alcock/fold_0
```

### Arguments

The script accepts several arguments to customize the training process. Here are the most important ones:

| Argument | Type | Default | Description |
|---|---|---|---|
| `--data`| str | (required) | Path to the training data directory (e.g., a specific fold). |
| `--output`| str | `./outputs`| Path to save model checkpoints and logs. |
| `--model`| str | `astromer_v1`| Name of the model architecture to use. |
| `--hidden`| int | `256`| Dimension of the hidden representations in the model. |
| `--d-model`| int | `256`| Alias for `--hidden` if used. |
| `--n-heads`| int | `4`| Number of attention heads in the Transformer encoder. |
| `--n-layers`| int | `4`| Number of layers in the Transformer encoder. |
| `--dropout`| float | `0.1`| Dropout rate for regularization. |
| `--epochs`| int | `100`| Number of training epochs. |
| `--bs`| int | `256`| Batch size for training. |
| `--lr`| float | `1e-4`| Learning rate for the optimizer. |
| `--patience`| int | `5`| Number of epochs with no improvement to wait before early stopping. |
| `--log-every`| int | `100`| How often (in steps) to log training metrics. |
| `--seed`| int | `42`| Random seed for reproducibility. |
| `--norm`| - | (flag) | If present, normalizes the input data. |
| `--debug`| - | (flag) | If present, runs in debug mode (usually with a smaller dataset). |

---

## Multi-GPU Pre-training

For larger datasets, pre-training can be significantly accelerated by using multiple GPUs. This project uses TensorFlow's `MirroredStrategy` for synchronous data-parallel training, which is handled by the `disttrain.py` script.

### Command

The script automatically detects and uses all available GPUs. The command is very similar to the single-GPU version, but you use the `disttrain.py` script instead.

```bash
python -m presentation.scripts.disttrain --data ./data/records/alcock/fold_0
```

### How It Works

The `disttrain.py` script wraps the model creation and training process within a `tf.distribute.MirroredStrategy` scope. This strategy replicates the model's variables on each available GPU. During each training step:
1. The input data batch is split evenly among the GPUs.
2. Each GPU computes the forward and backward pass on its slice of the data.
3. Gradients are summed across all GPUs (All-Reduce).
4. The synchronized gradients are used to update the model weights on all GPUs.

This ensures that the model on each GPU remains identical, effectively speeding up training by processing larger batches in parallel. The command-line arguments are the same as those for the single-GPU `pretrain.py` script.