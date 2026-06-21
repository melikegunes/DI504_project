# DI504 Pet Breed Classification

Melike Güneş — DI504 Foundations of Deep Learning

## Project scope


| Item           | Choice                                                                                                  |
| -------------- | ------------------------------------------------------------------------------------------------------- |
| **Task**       | 37-class pet breed classification                                                                       |
| **Dataset**    | [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/) via `torchvision.datasets.OxfordIIITPet` |
| **Baseline**   | Small custom CNN (trained from scratch)                                                                 |
| **Main model** | Pretrained **ResNet-18** (ImageNet weights)                                                             |
| **Metrics**    | Accuracy, macro F1, classification report, confusion matrix                                             |
| **Optional**   | Limited Optuna search (lr, batch size, weight decay)                                                    |




## Structure

```
504/
├── notebooks/di504_pet_project.ipynb
├── src/                    # data, models, train, evaluate
├── outputs/figures|metrics|checkpoints/
├── environment.yml
└── requirements.txt
```
