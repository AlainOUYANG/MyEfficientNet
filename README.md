# EfficientNet for cats-and-dogs classification

This is a simple example of using [EfficientNet](https://arxiv.org/abs/1905.11946) to perform a binary classification task.

Used the pretrained EfficientNet by [lukemelas/EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch).

Dataset is from [Dogs vs. Cats - Kaggle](https://www.kaggle.com/c/dogs-vs-cats). Due to limited computing power, a small subset is used for training:

- Training set: 101 cats and 101 dogs images
- Validation set: 100 cats and 100 dogs images
- Test set: 100 images

Part of the code is referenced from [this post](https://blog.51cto.com/AIXhao/2996748) of AIXhao.

Used Visdom to monitor the training loss, so before running `python main.py`, please run `python -m visdom.server` and open [http://localhost:8097](http://localhost:8097/).
