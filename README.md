# (RE-)PACRR Neural IR models

This is a Keras (TensorFlow) implementation of the model described in:

Kai Hui, Andrew Yates, Klaus Berberich, Gerard de Melo.
[PACRR: A Position-Aware Neural IR Model for Relevance Matching](https://arxiv.org/pdf/1704.03940.pdf).
*In EMNLP, 2017.*

Kai Hui, Andrew Yates, Klaus Berberich, Gerard de Melo.
[RE-PACRR: A Context and Density-Aware Neural Information Retrieval Model](https://arxiv.org/pdf/1706.10192.pdf).
*In Neu-IR workshop, 2017.*

Andrew Yates, Kai Hui.
[DE-PACRR: Exploring Layers Inside the PACRR Model](https://arxiv.org/pdf/1706.08746.pdf).
*In Neu-IR '17.*

Kai Hui, Andrew Yates, Klaus Berberich, Gerard de Melo.
[Co-PACRR: A Context-Aware Neural IR Model for Ad-hoc Retrieval](https://arxiv.org/pdf/1706.10192.pdf).
*In WSDM '18*

## Contact

***Code authors:*** [Kai Hui](https://people.mpi-inf.mpg.de/~khui/), [Andrew Yates](https://andrewyates.net/), and
[Abhinav Madahar](https://abhinavmadahar.com)

***Pull requests and issues:*** @khui @andrewyates

## Model overview

*(RE-)PACRR* models the interaction between a document and a query to evaluate
the relevance of the document to the query. The input is the similarity matrix
comparing the query and the document, and the output is a scalar. The full
pipeline of the model will be published soon.

## Getting Started

The code uses Python 2.

### Install Required Packages

First, we recommend installing Anaconda, which can be found
([here](https://www.continuum.io/downloads))

In addition, many of the required Python packages are available on pip, so it
is recommended that you install pip.

Instead of installing everything manually, you may want to use the `install`
script, which uses pip and is POSIX-compatible.

If you would prefer to install manually, please install these packages:

* **TensorFlow** ([instructions](https://www.tensorflow.org/install/))
* **Keras** ([instructions](https://keras.io/#installation))
* **numpy** and **scipy** ([instructions](https://www.scipy.org/install.html))
* **sacred** ([instruction](http://sacred.readthedocs.io/en/latest/quickstart.html#installation))
* **matplotlib** (available on pip)
* **pandas** (available on pip)
* **h5py** (available on pip)
* **pydot\_ng** (available on pip)

## Preparation: word2vec and similarity matrices

The model accepts a query-document matrix, so you need to supply them.
Fortunately, we already generated the matrices for clueweb as described in
(RE-)PACRR, which you can download from
[here](https://drive.google.com/file/d/0B3FrsWe6Y5YqdEtfSjI4N0h1LXM/view?usp=sharing)
and then extract using `tar xvf simmat.tar.gz`. This may take a few minutes.

If you would rather generate the matrices manually, please:

1. Extract the text from your WARC file

2. Get a pretrained word2vec corpus

3. Add vectors for the terms which are not included in the pretrained word2vec
   corpus. You can do so by running the `train_w2v.py` file, which adds new
   vectors and improves those already present.

4. Compute the similarity matrices for individual query-document pairs. This is
   not included in the code, so you will have to do so manually.

## Usage: train, predict and evaluation

The `set_env` script in the `bin` directory declares the environment variables
which are used in all other scripts in `bin`. If you want to replicate our
results, then don't worry about changing the variables in `set_env`. If you
want to explore the parameter space, then change something in `set_env` before
you try changing something elsewhere in the code.

![PHDComics: Dec 8, 2006](./phd120806s.gif)

### Train

To train the model, we recommend running the `train_model` script in bin. If
you would prefer to train directly, then run the command

```bash
python -m train_model with expname=$expname train_years=$train_years {param_name=param_val}
```

### Predict
We recommend that you use the `pred_per_epoch` script in bin to predict. If
you would prefer to train directly, then run the command

```bash
python -m pred_per_epoch with expname=$expname train_years=$train_years test_year=$test_year {param_name=param_val}
```

### Evaluation

You can evaluate the model's prediction over the three benchmarks as described
in our RE-PACRR paper.

Before evaluating, make sure that you dumped [all the adhoc
trec-runs](http://trec.nist.gov/results/) from 2009 through 2014 into
`utils/data/trec_run` and their corresponding evaluation results into
`utils/data/eval_trec_runs`. These are needed for Rerank-ALL.

Our evaluation code generates the exp that we reported in our PACRR and REPACRR
papers, but you can develop a different evaluation pipeline to use a different
dataset or benchmark.

For example, the usage of the six years' Web Track for training/validation/test
is hard coded in the `train_test_years` in utils/config.py, and one needs to
edit it when fewer years are used.

To evaluate the model, you can run the `evals` script in bin or run the following commands:

```bash
python -m evals.docpairs with expname=$expname train_years=$train_years {param_name=param_val}
python -m evals.rerank with expname=$expname train_years=$train_years {param_name=param_val}
```

## File Structure Overview
`bin/` contains the binaries and logical binaries which can be executed to
train, evaluate, and otherwise use the model.

`data/` contains trec web data.

`evals/` contains the scripts which evaluate the model.

`libs/` contains libraries which our code uses and which we didn't write.

`models/` contains the code which defines the PACRR model and its extensions
(e.g. RE-, CO-).

`pacrr/` is not included by default in the repository, but is built when
training the model. It contains the files which save a PACRR model instance as
well as misc. information about it, like loss and plots. To change the name of
the directory, set `$parentdir` in `bin/set_env` to be the new name of the
directory. Make sure to move the files from the original directory to the new
one.

`preprocess/` contains various libraries for preprocessing the data. We didn't
write this either.

`simmat/` is not included by default in the repository, but the code expects it
to exist and contain the similarity matrices, so you should put them here.

`test/` contains testing files.

`utils/` contains various utility functions which we wrote.

## Citation

If you use the code, please cite the following papers:

### PACRR

```latex
@inproceedings{hui2017pacrr,
	title={PACRR: A Position-Aware Neural IR Model for Relevance Matching},
	author={Hui, Kai and Yates, Andrew and Berberich, Klaus and de Melo, Gerard},
	booktitle={Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing},
	pages={1060--1069},
	year={2017}
}
```

### RE-PACRR

```latex
@inproceedings{hui2017re,
	title={RE-PACRR: A Context and Density-Aware Neural Information Retrieval Model},
	author={Hui, Kai and Yates, Andrew and Berberich, Klaus and de Melo, Gerard},
	booktitle={The SIGIR 2017 Workshop on Neural Information Retrieval},
	year={2017}
}
```

### DE-PACRR

```latex
@inproceedings{hui2017de,
	title={DE-PACRR: Exploring Layers Inside the PACRR Model},
	author={Hui, Kai and Yates, Andrew},
	booktitle={The SIGIR 2017 Workshop on Neural Information Retrieval},
	year={2017}
```

### CO-PACRR

```latex
@inproceedings{hui2017co,
	title={Co-PACRR: A Context-Aware Neural IR Model for Ad-hoc Retrieval},
	author={Hui, Kai and Yates, Andrew and Berberich, Klaus and de Melo, Gerard},
	booktitle={Proceedings of the Eleventh ACM International Conference on Web Search and Data Mining},
	year={2017}
```
