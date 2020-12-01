# Hybrid DNN-HMM model for isolated digit recognition
Python implementation of a hybrid DNN-HMM models for isolated digit recognition.

Forced alignments are obtained from a GMM-HMM model and used to train the DNN.
The DNN is a simple multi-layer perceptron (MLP) implemented using scikit-learn.

### How to run

```
python3 submission.py <opt-args> train test
```

* `train` is the training data
* `test` is the test data

The optional arguments are:
* `--mode`: Type of model (`mlp`, `hmm`). Default: `mlp`
* `--niter`: Number of iterations to train the HMM. Default = 10
* `--nstate`: Number of states in HMM model. Default = 5
* `--nepoch`: Maximum number of epochs for training the MLP. Default=10
* `--lr`: Learning rate for the MLP. Default=0.01
* `--debug`: Uses only top 100 utterances for train and test

### Training data format

I cannot upload the full training and test data (for copyright reasons), but a small sample of the training data can be found at this [Google Drive link](https://drive.google.com/file/d/1NhF7fuX54jau9iXxuitOfm9QRQPHNW2Q/view?usp=sharing). This should help in understanding the format of the data.

### Help

This code is based on a template provided by Shinji Watanabe (Johns Hopkins University), written for a course project.

For assistance, contact `draj@cs.jhu.edu`.
 
