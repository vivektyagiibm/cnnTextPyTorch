## CNN Text Classifier PyTorch
State of the art single convolution layer + maxpool + fully connected last layer, Convolution Text Classifier with word2vec embeddings as input. Adding more convolution layers did not improve the performance on Moview review (SST-2) and other datasets and hence the single convolution layer architecture has been chosen. 

Trained and tested on "Movie Review Dataset". 

Purpose is to provide a tutorial to the interested user on how to train a state of the art CNN text classifier using user-provided embeddings (word2vec in this case). 
A classical SGD is applied to also exemplify the updates in a deep learning system instead of the optimizer method. 

## Requirement
* Download GoogleNews-vectors-negative300.bin to the same directory where you clone this repo.
* python 3
* pytorch > 0.1
* numpy

<embed src="https://github.com/vivektyagiibm/cnnTextPyTorch/blob/master/cnn.pdf" width="800px" height="2100px" />

<img src="https://github.com/vivektyagiibm/cnnTextPyTorch/blob/master/cnn.pdf" alt="HMTL Architecture" width="350"/>

Tokenize and clean your Movie Review (SST-2) data (already in this repo). This dumps a pickle file with the train and tets datasets. 
## Usage
```
python process_data.py GoogleNews-vectors-negative300.bin 
```

Run the CNN Text classifier model training
```
python extractMRWord2VecNumpy_CNNTextClassifier_Train.py > log.txt & 
```

You can examine the log.txt to see how the train and validation error changes over batches and epochs. 

Run the testModel.py

```
python testModel.py 
```
## Result

You will get 80.74% test accuracy on the 10% held  out test-set


