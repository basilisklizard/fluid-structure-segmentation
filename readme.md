## Fluid-structure segmentation

### 1. Project description 

The goal of this project was to design an application that can perform **image segmentation** on frames obtained from **particle image velocimetry (piv)** experiments. More precisely, the segmentation process enables labeling of fluid and non-fluid portions within a given image. The resulting labeling/segmentation is required for the precise computation of velocity and pressure fields. 

The target piv-data contains fish (non-fluid) swimming against a flow of oncoming water (fluid). This represents a challenge because of the constantly changing body shape during experiments. 

The idea in this project was to train a neural network to learn how to distinguish between fluid portions, represented by tiny particles in the flow, and non-fluid portions. For this purpose, a combination of an **artificially generated training dataset** together with **pre-labeled data** was used for training. 

Two neural network architectures are implemented:

* Autoencoder
* U-net

The autoencoder was implemented based on ideas from Venneman and Rösgen, 2020 (https://link.springer.com/article/10.1007/s00348-020-02984-w). However, the U-net architecture performed significantly better for the targeted datasets. 

### 2. How to install the project

TBD

Add repository to your python path:

```export PYTHONPATH="<PATH_TO_REPOSITORY>:$PYTHONPATH"```

### 3. How to run the project and evaluate the results

TBD
