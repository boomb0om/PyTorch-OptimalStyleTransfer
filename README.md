# PyTorch-OptimalStyleTransfer
This is an **unofficial** PyTorch implementation of paper [A Closed-form Solution to Universal Style Transfer - ICCV 2019](https://arxiv.org/abs/1906.00668)

1. [Colab demo](https://colab.research.google.com/drive/1Y8epP-g2MMGI1Ri-lvOttIO_ebdIXORA?usp=sharing) <a href="https://colab.research.google.com/drive/1Y8epP-g2MMGI1Ri-lvOttIO_ebdIXORA?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>
2. [Official implementation](https://github.com/lu-m13/OptimalStyleTransfer) (Lua)

Some examples:![](https://raw.githubusercontent.com/boomb0om/PyTorch-OptimalStyleTransfer/main/imgs/example_results.png)

Comparison with other methods:![](https://raw.githubusercontent.com/boomb0om/PyTorch-OptimalStyleTransfer/main/imgs/comparison_small.png)

## Installation

1. Install packages
   `pip install -r requirements.txt`

2. Download [model weights](https://drive.google.com/file/d/1-gBUEnJ1Wdqd9naAnfj2R-tYoS3vW6M8/view?usp=sharing) and extract archive into `models/` folder

## Style Transfer

You can test model in [demo.ipynb](https://github.com/boomb0om/PyTorch-OptimalStyleTransfer/blob/main/demo.ipynb) or use a script:

`python test.py --content=content/brad_pitt.jpg --style=style/picasso_self_portrait.jpg `



