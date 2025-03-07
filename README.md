# Coffee Beans Defect Classification

This project implements deep learning models for classifying 19 types of coffee bean defects using PyTorch.

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

## Dataset Classes

The dataset contains **19 classes of coffee bean defects** based on the Indonesian National Standard (SNI). Below is a detailed explanation of each class:

| **No.** | **Class Name**           | **Description**                                                                 |
|---------|--------------------------|---------------------------------------------------------------------------------|
| 1       | Batu besar               | Large stones mixed with coffee beans.                                           |
| 2       | Batu kecil               | Small stones mixed with coffee beans.                                           |
| 3       | Batu sedang              | Medium-sized stones mixed with coffee beans.                                    |
| 4       | Berjamur                 | Moldy beans caused by fungal growth.                                            |
| 5       | Coklat                   | Brown beans, often indicating over-fermentation or improper drying.             |
| 6       | Gelondong                | Whole coffee cherries or foreign fruit material mixed with beans.               |
| 7       | Hitam                    | Fully black beans, often due to over-fermentation or fungal infection.          |
| 8       | Hitam pecah              | Broken black beans, indicating physical damage and fermentation issues.         |
| 9       | Hitam sebagian           | Partially black beans, showing uneven fermentation or fungal infection.         |
| 10      | Kulit besar              | Large pieces of parchment or husk mixed with beans.                             |
| 11      | Kulit kecil              | Small pieces of parchment or husk mixed with beans.                             |
| 12      | Kulit sedang             | Medium-sized pieces of parchment or husk mixed with beans.                      |
| 13      | Lubang 1                 | Beans with a single hole, often caused by insect damage.                        |
| 14      | Lubang lebih dari 1      | Beans with multiple holes, indicating severe insect damage.                     |
| 15      | Muda                     | Immature beans, harvested before reaching full maturity.                        |
| 16      | Pecah                    | Broken beans, often due to improper handling or processing.                     |
| 17      | Ranting besar            | Large twigs or branches mixed with beans.                                       |
| 18      | Ranting kecil            | Small twigs or branches mixed with beans.                                       |
| 19      | Ranting sedang           | Medium-sized twigs or branches mixed with beans.                                |

## Dataset Source

The dataset used in this project is publicly available for download. You can access it using the following links:

- **Google Drive**: [Download Coffee Bean Defect Dataset on GDrive](https://drive.google.com/drive/folders/149SxkXrlo7FChDDLLtvFs6y_MoFEDTYs?usp=sharing)

## Training

Train a model using the following command:

> Don't forget to rename the dataset folder into `dataset_kopi`

```bash
python train.py --model efficientnet --batch_size 32 --lr 0.001 --epochs 30
```

Available models:
- efficientnet
- resnet50
- mobilenetv3
- densenet121
- vit
- convnext
- regnet

## Weights & Biases Integration

This project uses Weights & Biases for experiment tracking. Each run is automatically named using a timestamp in the format `model_YYYYMMDD-HHMM`.

To disable wandb logging, use the `--no-wandb` flag:

```bash
python train.py --model efficientnet --no-wandb
```

To view your experiment results, go to the Weights & Biases dashboard at [wandb.ai](https://wandb.ai).
