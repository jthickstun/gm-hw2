# CSE 599i (Generative Models) Homework 2 #

In this part of Homework 2, your will implement several variants of the VAE, and run experiments using the MNIST and Binarized MNIST datasets.

You can download the datasets [__here__](https://courses.cs.washington.edu/courses/cse599i/20au/resources/data.tar.gz) (extract them to the `data/` directory in the root of this repository). I've provided code for loading and processing this data into minibatches in `mnist.py` and `bmnist.py`.

Scaffolding for your VAE models is provided in `models.py`, and you will implement the loss functions for various types of VAE's in `losses.py`.

Framework code for training your models is found in `gaussian_vae.ipynb`, `pixelcnn.ipynb`, and `binaryvae`. I recommend using Google Colab to execute these notebooks (with a GPU accelerator attached). For debugging, you may find it helpful to modify the default hyper-parameters to build smaller models that are faster to train.
