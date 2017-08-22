# (RE-)PACRR Neural IR models 

This is a Keras (TensorFlow) implementation of the model described in:

Kai Hui, Andrew Yates, Klaus Berberich, Gerard de Melo.
[PACRR: A Position-Aware Neural IR Model for Relevance Matching](https://arxiv.org/pdf/1704.03940.pdf).
*In EMNLP, 2017.*
Kai Hui, Andrew Yates, Klaus Berberich, Gerard de Melo.
[RE-PACRR: A Context and Density-Aware Neural Information Retrieval Model](https://arxiv.org/pdf/1706.10192.pdf).
*In Neu-IR workshop, 2017.*


## Contact
***Code author:*** Kai Hui and Andrew Yates

***Pull requests and issues:*** @khui @andrewyates

## Contents
* [Model Overview](#model-overview)
* [Getting Started](#getting-started)
    * [Install Required Packages](#install-required-packages)
    * [Download Pretrained Models on ClueWeb](#download-pretrained-models-clueweb)

## Model overview

The *(RE-)PACRR* models the interaction between a query-document pair, evaluating the
relevance of the document for the query. The input is the similarity matrix between 
a query and a document, and the output is a scaler. The full pipeline of the model will be 
published soon. 

## Getting Started

### Install Required Packages

First install with Anaconda (Recommended)
* **Anaconda** ([instructions](https://www.continuum.io/downloads))
Thereafter install tensorflow and keras on Anaconda.
* **TensorFlow** ([instructions](https://www.tensorflow.org/install/))
* **Keras** ([instructions](https://keras.io/#installation))


### Download Pretrained Models on ClueWeb
