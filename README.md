# GPTune: Finetuned GPT-2 Model Repository

GPTune attempts to make finetuning the recently released GPT-2 model easier by providing some good presets.

## How to Use

### Training

In terminal, run
```
python gptune.py DATASET_NAME --run_name MODEL_NAME_GIVEN_BY_YOU --mode train
```

There are other parameters; right now for training an example that will work well for the 774M model on a 16GB RAM single-GPU setting is:

```
python gptune.py --dataset DATASET_NAME --run_name ddbios2 --optimizer adam --only_train_transformer_layers --truncate_training 128
```

### Testing

#### Generating Unconditional Samples

In terminal, run
```
python gptune.py --dataset DATASET_NAME --run_name MODEL_NAME_GIVEN_BY_YOU --mode test --sample_num 2
```

You can change sample_num to a different number.

#### Interactive Conditional Samples

```
python gptune.py --dataset DATASET_NAME --run_name harry --mode interactive --sample_num 1
```

For more settings you can look at the source code (specifically, gptune.py).

## Pretrained Models

GPTune currently exists to provide a number of pretrained models for people to plug&play with.

### 117M Models:

The following models are for the 117M parameter model.

* [Custom Poetry Collection](https://drive.google.com/file/d/1w3fNoQJcJCVlouxQbTpef2IKNdi7BpgF/view?usp=sharing)
* [D&D Biographies](https://drive.google.com/file/d/1qBxIX_V3uXoTY24BLJHlqSpk2m3mgLzd/view?usp=sharing) (Replication of [Janelle Shane](https://twitter.com/JanelleCShane)'s Excellent Work)
* [Magic: The Gathering Cards](https://drive.google.com/file/d/1HP5DssYWR_9Io2yLdP6Qm1PtNwteFJp3/view?usp=sharing)
* [Plot Summaries](https://drive.google.com/file/d/1U8tf76BvUbXv2vAelG3qKOEwTPhyVmkd/view?usp=sharing)
* [Science Fiction Stories](https://drive.google.com/file/d/1mfmEoTW1b-Wo7r6EmcGTCRb-3Wp6QMN4/view?usp=sharing)

### 774M Models:

Coming soon!

### Acknowledgements

GPTune is based on [the finetuning code released by nshepperd](https://github.com/nshepperd/gpt-2/tree/finetuning).

# Original README

# gpt-2

Code and samples from the paper ["Language Models are Unsupervised Multitask Learners"](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf).

For now, we have only released a smaller (117M parameter) version of GPT-2.

See more details in our [blog post](https://blog.openai.com/better-language-models/).

## Installation

Git clone this repository, and `cd` into directory for remaining commands
```
git clone https://github.com/openai/gpt-2.git && cd gpt-2
```

Then, follow instructions for either native or Docker installation.

### Native Installation

Download the model data
```
sh download_model.sh 117M
```

The remaining steps can optionally be done in a virtual environment using tools such as `virtualenv` or `conda`.

Install tensorflow 1.12 (with GPU support, if you have a GPU and want everything to run faster)
```
pip3 install tensorflow==1.12.0
```
or
```
pip3 install tensorflow-gpu==1.12.0
```

Install other python packages:
```
pip3 install -r requirements.txt
```

### Docker Installation

Build the Dockerfile and tag the created image as `gpt-2`:
```
docker build --tag gpt-2 -f Dockerfile.gpu . # or Dockerfile.cpu
```

Start an interactive bash session from the `gpt-2` docker image.

You can opt to use the `--runtime=nvidia` flag if you have access to a NVIDIA GPU
and a valid install of [nvidia-docker 2.0](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)).
```
docker run --runtime=nvidia -it gpt-2 bash
```

## Usage

| WARNING: Samples are unfiltered and may contain offensive content. |
| --- |

Some of the examples below may include Unicode text characters. Set the environment variable:
```
export PYTHONIOENCODING=UTF-8
```
to override the standard stream settings in UTF-8 mode.

### Unconditional sample generation

To generate unconditional samples from the small model:
```
python3 src/generate_unconditional_samples.py | tee /tmp/samples
```
There are various flags for controlling the samples:
```
python3 src/generate_unconditional_samples.py --top_k 40 --temperature 0.7 | tee /tmp/samples
```

To check flag descriptions, use:
```
python3 src/generate_unconditional_samples.py -- --help
```

### Conditional sample generation

To give the model custom prompts, you can use:
```
python3 src/interactive_conditional_samples.py --top_k 40
```

To check flag descriptions, use:
```
python3 src/interactive_conditional_samples.py -- --help
```

### Fine tuning on custom datasets

To retrain GPT-2 117M model on a custom text dataset:

```
PYTHONPATH=src ./train --dataset <file|directory|glob>
```

If you want to precompute the dataset's encoding for multiple runs, you can instead use:

```
PYTHONPATH=src ./encode.py <file|directory|glob> /path/to/encoded.npz
PYTHONPATH=src ./train --dataset /path/to/encoded.npz
```

## GPT-2 samples

| WARNING: Samples are unfiltered and may contain offensive content. |
| --- |

While we have not yet released GPT-2 itself, you can see some samples from it in the `gpt-2-samples` folder.
We show unconditional samples with default settings (temperature 1 and no truncation), with temperature 0.7, and with truncation with top_k 40.
We show conditional samples, with contexts drawn from `WebText`'s test set, with default settings (temperature 1 and no truncation), with temperature 0.7, and with truncation with top_k 40.

## Citation

Please use the following bibtex entry:
```
@article{radford2019language,
  title={Language Models are Unsupervised Multitask Learners},
  author={Radford, Alec and Wu, Jeff and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya},
  year={2019}
}
```

## Future work

We may release code for evaluating the models on various benchmarks.

We are still considering release of the larger models.

## License

MIT
