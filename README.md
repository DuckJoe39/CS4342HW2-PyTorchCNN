# CS4343 HW2: Custom CNN with Skip Connections (CIFAR-10)

A custom convolutional neural network with residual (skip) connections, implemented from scratch in PyTorch and trained on CIFAR-10. The model does not use any prebuilt network from `torchvision.models`.

## Contents

| File | Description |
|:--|:--|
| `CS4343_HW2.ipynb` | Notebook with the full implementation, training, and evaluation |

## Model Architecture

| Component | Structure | Output Size |
|:--|:--|:--|
| Stem | 3x3 Conv (32 ch) + BatchNorm + ReLU | 32 x 32 x 32 |
| Stage 1 | 2 skip-connection blocks, 32 channels | 32 x 32 x 32 |
| Stage 2 | 2 skip-connection blocks, 64 channels (first block stride 2) | 64 x 16 x 16 |
| Stage 3 | 2 skip-connection blocks, 128 channels (first block stride 2) | 128 x 8 x 8 |
| Pooling | Global average pooling | 128 x 1 x 1 |
| Classifier | Linear | 10 logits |

Each skip-connection block computes `Y = ReLU(F(X) + S(X))`:

- `F(X)`: Conv-BN-ReLU followed by Conv-BN (3x3 kernels)
- `S(X)`: identity when shape is unchanged; otherwise a 1x1 Conv (same stride as the main path) followed by BatchNorm

Reusable modules implemented in the notebook: `ConvBNReLU`, `SkipConnectionBlock`, `CustomCNN`.

## Data Pipeline

- Dataset: CIFAR-10 (50,000 train / 10,000 test images, 10 classes)
- Reproducible split of the training set (seeded `torch.randperm`): **45,000 train / 5,000 validation**
- Training transforms: `RandomCrop(32, padding=4)`, `RandomHorizontalFlip()`, `ToTensor()`, `Normalize`
- Validation / test transforms: `ToTensor()`, `Normalize` (no augmentation)

## Training Setup

- Loss: cross-entropy
- Optimizer: SGD, momentum 0.9, Nesterov, weight decay
- Scheduler: `CosineAnnealingLR`
- Initialization: He normal (`fan_out`) for convolutions, BatchNorm scale 1 / shift 0, Linear weights ~ N(0, 0.01^2) with zero bias
- The best-validation-accuracy weights are restored at the end of training
- Final training: 50 epochs on the full 45,000-image training split

### Hyperparameter Search

Four configurations were compared with short runs on a subset of the data (10,000 train / 2,000 validation images). The test set was not used for selection.

| Config | Learning Rate | Weight Decay |
|:--|--:|--:|
| 1 | 0.03 | 5e-4 |
| 2 | 0.05 | 5e-4 |
| 3 | 0.10 | 5e-4 |
| 4 | 0.10 | 1e-4 |

Selected configuration: **TODO (fill in `lr` / `weight_decay` of `best_config`)**

## Results

| Metric | Value |
|:--|--:|
| Best validation accuracy | TODO |
| Test loss | TODO |
| Test accuracy | TODO |

Training and validation curves (loss, accuracy, learning-rate schedule) are included in the notebook output.

## How to Run

```bash
pip install torch torchvision matplotlib jupyter
jupyter notebook CS4343_HW2.ipynb
```

Restart the kernel and run all cells from top to bottom. CIFAR-10 is downloaded automatically to `./data` on first run (about 170 MB, not tracked by Git). A GPU is recommended for the 50-epoch final training.

## Requirements

- Python 3.12
- PyTorch, torchvision
- NumPy, Matplotlib
