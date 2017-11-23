# Project Milestone

## Generic Goals for next Thursday: 

- You have your research question(s) clear
- You have read and understood all relevant literature
- You have results ready that could function as your baseline

For those comparing different language models:
- Understand how the data is commonly handled for RNN-style models, see below.
- Have KenLM perplexities on the test set
- Bengio LM implementation working
- Bengio LM perplexity results on test set 
- You can run PyTorch RNN language models; you understand the command line options and how the data set is divided using the -bptt option.
- For those of you implementing RAN: initial implementation done, perhaps not yet with results.

### KenLM
Make sure you have figured out how to use KenLM and get a perplexity on the test set.
We recommend that you use kenlm from the command line. You can first try training a 3-gram model, and then go towards 4 and 5-gram models

This is how to train a 3-gram model: 

```bash
./lmplz -o 3 < train.txt > train.arpa
```

You will see that KenLM complains about the <unk> tokens in the dataset. You can work around this by replacing them with “UNK”, like this:

```bash
cat train.txt | sed "s/<unk>/UNK/g" > train.txt.UNK
cat test.txt | sed "s/<unk>/UNK/g" > test.txt.UNK
```

Now you can train using the above command, and test like this:

```bash
cat test.txt.UNK | ./query train.arpa
```

You should get a test perplexity of about 142.7 using this approach with a 4-gram model.
For your report it would be nice to know how models with different orders perform (bi-, tri-, 5-gram).

A possibly better/different way to handle the unknown words is to use KenLM with a restricted vocabulary, and to exclude UNK from this vocabulary. You will have to see if this results in a better perplexity or not. See lmplz --help on how to do this.

### Bengio NNLM:

Try to make final adjustments to this code so that you can evaluate it on the same data set.
You can take inspiration from PyTorch RNN language modeling code. In particular, take a look at how it handles data: https://github.com/pytorch/examples/blob/master/word_language_model/main.py 

You can see that in Neural Language Modeling it is the tradition to treat the training set as one long sequence of words, and to then cut this long sequence into parts of a certain size BPTT (e.g. BPTT=35 words). To make this work, you will have to add an end-of-sequence symbol `<eos>` after each sentence.

For example, if your training set is this:

```
A b c d 
E f g h
```

And you are using BPTT=2, then you would train your RNN on the following sequences:

```
A b
c d
<eos> E
f g
h <eos>
```

For Bengio’s LM, this is less of an issue, since you are already limited the history for 
your predictions to N-1. For positions where your history is shorter than N-1, you 
can input a padding vector (a special word embedding e.g. for the word “`*PAD*`”).

If your training is slow, you can do the following:

- Use the pre-trained Glove word embeddings, and do not train them (feed them as input).
- Use smaller hidden layers
- Use mini-batches. Processing multiple sentences (or predictions) at a 
time speeds up training a LOT. But it also makes things more complicated, 
because you add an extra dimension to your tensors (the first dimension, the batch dimension). 

You can find a simple example of mini-batching in the solutions to the pytorch tutorial: 
https://github.com/tdeoskar/NLP1-2017/blob/master/pytorch-tutorial/solutions/deep-cbow-minibatch.py 
and you can also look at the “batchify” method in the PyTorch RNN code. (see link above).

### Other language models:

If you want to verify if your implementation is correct, consider creating an 
artificial data set where it is very easy to predict the next word.

For example, train on:

```
a b c d e
b c d 
etc.
```

(you could also make a short python program that generates such sequences).

Then test on e.g.

```
c d e
a b c  
etc.
```

Your network should figure out very quickly how to do this, 
and should get perfect perplexity on your train and test set.

If there are certain things where you would need more explanation, 
and you think everyone would benefit from that, please send us a short and concise e-mail about it, 
so that we can prepare an explanation for everyone for the next lab session, or post it on Piazza (in the project folder).


## Tips

### Pre-trained Embeddings

PyTorch automatically considers each instance of `torch.nn.Embedding` as a parameter that needs training. So even if you load in pre-trained embeddings like described above, PyTorch will alter them during training. If you don’t want PyTorch to alter them during training (this will save your computer much computation time!) you must do the following.

Let `embedding` be the instance of the `torch.nn.Embedding` layer in which you uploaded the pre-trained embeddings. Let `model` be the instance of your model class of which `embedding` is a part. Then you should do:

```python
embedding.weight.requires_grad = False
parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.SGD(parameters, lr=learning_rate)
```

### Cross-entropy loss

Note that `torch.nn.CrossEntropyLoss` is a combination of `LogSoftMax` and `NLLLoss`.
This means that you should not use it on output that has already gone through a softmax! 

Either:

- Use a `softmax` at the end of your forward pass
- Then use `NLLLoss` (negative log-likelihood)

Or:
- Only use `torch.nn.CrossEntropyLoss`, but don't take a softmax. You don't need a softmax to identify the argmax (the prediction), since the maximum value before softmax will still be the maximum value afterwards.
