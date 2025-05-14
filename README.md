# Exploring Selective Layer Fine-Tuning in Federated Learning

This is the implementation of our paper "[Exploring Selective Layer Fine-Tuning in Federated Learning](https://arxiv.org/abs/2408.15600)".

## Installation instructions

Install Python environment with conda:

```bash
conda create -n fedsel python=3.9 -y
conda activate fedsel
```

Then install the Python packages in `requirements.txt`:

```bash
pip3 install -r requirements.txt
```
NOTE: you may need to check the version of some packages such as torch and transformers.

### Download the models
The CLIP and BERT models need to be downloaded from the Hugging Face to the `../models/` directory.
Or you can set the MODEL_PATH to official HF name of this model to enable the automatic download of the models.

### Download the datasets

For CIFAR-10:
```shell
python src/data_helpers/prepare_cifar.py --dataset=cifar10 --clip --client_num_in_total=100 --alpha=0.1
```

For DomainNet and XGLUE:
You can download the datasets from the official website and put them in the `../data/` directory. These datasets will be prepared automatically when running the experiment.

## Run an experiment 

```shell
python src/server.py --dataset=[Dataset name] --model_type=[Model name] --strategy=[Strategy name] --n_layers=[Min_n_layers] --n_layers_inc=[Max_n_layers]
```

| Argument       | Description   | Choices                              |
|----------------|---------------|--------------------------------------|
| `dataset`    | The name of the dataset  |  cifar10, domainnet, xglue  |
| `model_type` | The name of the model  | clip, xlm-roberta-large, roberta-large, xlm-roberta-base, roberta-base |
| `strategy`   | The layer selection strategy   | full, pro, top, bottom, both, sgn, rgn   |
| `n_layers`   | The number of minimal selected layers in each client        | 1, 2 (Integer)      |
| `n_layers_inc`| The number of maximal selected layers in each client       | 0, 4 (Integer)      |


For example, run the proposed method:

```shell
python src/server.py --dataset=domainnet --model_type=clip --strategy=pro --n_layers=1 --n_layers_inc=4
```

## Citing

If you use this code in your research or find it helpful, please consider citing our paper:
```
@inproceedings{sun2025exploring,
  author    = {Sun, Yuchang and Xie, Yuexiang and Ding, Bolin and Li, Yaliang and Zhang, Jun},
  title     = {Exploring Selective Layer Fine-Tuning in Federated Learning},
  booktitle = {Proc. IEEE International Symposium on Information Theory (ISIT)},
  address   = {Michigan, USA},
  month     = {June},
  year      = {2025},
}
```

## Contact

If you have any questions, please feel free to contact us via hiyuchang@outlook.com.

## Acknowledgements
The initial implement of this repo is based on the [pFedLA](https://github.com/KarhouTam/pFedLA). We thank the authors for their contribution.
