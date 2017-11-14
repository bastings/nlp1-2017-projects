# Neural Language Modeling

## Report requirements

As a final product, you will be asked to write a report with your findings, which should at least contain:

- A background section, in which you explain what is language modelling, what kind of language models you studied, and what is the problem you focus on. To write this background section, you can use the suggested papers, but also add papers;
- A summary of the relevant literature
    - A description of your replication;
- A formal discription of your model(s) and changes/improvements you made compared to the literature
- Analysis of your results
- A conclusion

## Notes after 9 nov lab

 - You can find the PyTorch code example, with the Neural n-gram LM, here: http://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html
 
 - You can find an RNN LM PyTorch code example here (advanced!): https://github.com/pytorch/examples/tree/master/word_language_model 
  
 - As your data set, for English, please use the Penn Treebank Dataset, so that we can compare perplexities: https://github.com/pytorch/examples/tree/master/word_language_model/data/penn
  
 - If you're interested in how to get word embeddings **without** neural networks, then check out this blog post: http://multithreaded.stitchfix.com/blog/2017/10/18/stop-using-word2vec/

## Introduction

Language modeling is a central problem in NLP, since it is so useful for many tasks, e.g. in machine translation. 
A language model gives the probability of the next word given a history of previous words, 
and by doing so it can assign a probability to a whole sentence. 

In this project you will implement your own language model(s) in PyTorch, and you will compare their perplexities on test data.

Firstly, you will take an n-gram language model approach to have a baseline. 
For this part, you can use [KenLM](http://kheafield.com/code/kenlm/) and estimate various 
language models of different orders (i.e. taking more and more history into account). 
Note that KenLM implements Kneser-Ney smoothing (J&M explains it).

Then, you will implement a neural language model. There are different ones you can try, if time permits. 
Make sure that you at least implement the first one.

To make training of neural networks faster, you may want to use pre-trained word embeddings, so that you do not have to train so many parameters. You can download the 50-d word vectors from https://nlp.stanford.edu/projects/glove/ 

1. Bengio NNLM. This is a feed-forward network that predicts the next word given a fixed amount of previous words.
2. Simple RNN LM. You can also train a simple RNN to predict the next word given the previous word and the hidden state of the RNN. Since this is a simple RNN, in practice it will not remember a lot of history, but it will be nice for comparison reasons, and it is relatively easy to train.
3. Recurrent Additive Networks (RAN) RANs are a simpler version of LSTMs. This makes them not only faster to train, it also allows us to inspect which previous word was most influential in predicting the current word. Using RANs can be cool for analysis, and you can try to tweak them to get even better performance.
4. LSTM / GRU.  You can also train an LSTM (or GRU) language model. See the previous paper for details. Note that, of all networks, training this will take the longest, especially on CPU.

## Required reading

1. J&M [chapters n-gram Language Models and Neural Language Models](https://web.stanford.edu/~jurafsky/slp3/)
2. [Bengio (2003) NNLM](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
3. [Mikolov (2010) RNN-LM](http://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf)
4. [Lee et al. (2017) RAN](https://arxiv.org/abs/1705.07393)
