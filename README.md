## Retrieval-Based Conversational Model in Tensorflow (Ubuntu Dialog Corpus)

#### [Please read the blog post for this code](http://www.wildml.com/2016/07/deep-learning-for-chatbots-2-retrieval-based-model-tensorflow)

#### Overview

The code here implements the Dual LSTM Encoder model from [The Ubuntu Dialogue Corpus: A Large Dataset for Research in Unstructured Multi-Turn Dialogue Systems](http://arxiv.org/abs/1506.08909).

#### Setup

This code uses Python 3 and Tensorflow >= 0.9. Clone the repository and install all required packages:

```
pip install -U pip
pip install numpy scikit-learn pandas jupyter
```

#### Get the Data


Download the train/dev/test data [here](https://drive.google.com/open?id=0B_bZck-ksdkpVEtVc1R6Y01HMWM) and extract the acrhive into `./data`.


#### Training

```
python udc_train.py
```


#### Evaluation

```
python udc_test.py --model_dir=...
```


#### Evaluation

```
python udc_predict.py --model_dir=...
```
