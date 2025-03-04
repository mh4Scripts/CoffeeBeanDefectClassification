# Coffee Classification

This project implements deep learning models for coffee bean classification using PyTorch.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare your dataset in the following structure:
```
dataset_kopi/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
├── class2/
│   ├── image3.jpg
│   ├── image4.jpg
...
```

## Training

Train a model using the following command:

> Don't forget to rename the dataset folder into `dataset_kopi`

```bash
python train.py --model efficientnet --batch_size 32 --lr 0.001 --epochs 30
```

Available models: `efficientnet`, `shufflenet`, `resnet152`, `vit`

## Weights & Biases Integration

This project uses Weights & Biases for experiment tracking. Each run is automatically named using a timestamp in the format `model_YYYYMMDD-HHMM`.

To disable wandb logging, use the `--no-wandb` flag:

```bash
python train.py --model efficientnet --no-wandb
```

To view your experiment results, go to the Weights & Biases dashboard at [wandb.ai](https://wandb.ai).
