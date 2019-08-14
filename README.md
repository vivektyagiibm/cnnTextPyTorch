## CNN Text Classifier PyTorch
A simple one convolution layer + maxpool + full connected  Convolution Text Classifier with word2vec embeddings as input. 
Trained and tested on "Moview Review Dataset". 

Purpose is to provide a tutorial to the interested user on how to train a state of the art CNN text classifier using user-provided emebddings (word2vec in this case). 
A classical SGD is applied to also exemplify the updates in a deep learning system instead of the optimizer method. 


Download GoogleNews-vectors-negative300.bin to the same directory where you clone this repo.

Tokenize and clean your Movie Review (SST-2) data (already in this repo). This dumps a pickle file with the train and tets datasets. 
## Usage
```
python process_data.py GoogleNews-vectors-negative300.bin 
```

Run the CNN Text classifier model training
```
python extractMRWord2VecNumpy_CNNTextClassifier_Train.py > log.txt & 
```

You can examine the log.txt to see how the train and validation error chanages over epochs. 

Run the testModel.py

```
python testModel.py 
```
## Result

You will get 80.74% test accuracy on the 10% held  out test-set


