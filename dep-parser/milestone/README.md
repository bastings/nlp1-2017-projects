# Project milestone

Below you will see detailed descriptions of milestones for your project. We also included some tips for each of the parts.

:warning: Read this through carefully :warning:

## Dependency-data handling

**Milestone:** You have finished the code that takes care of reading in and writing out text from the CONLL-U file type. This code will do all the data processing. You will need it both during training as well as prediction time.

### Tips

* The CONLL-U file does not have an `<unk>` word that represents unknown words. You do need to train good vector representations for `<unk>` though because during evaluation on the test-set you will encounter many unknown words. A possible idea is to replace all the words in your training file that occur just once with the word `<unk>`.
    * To give you an impression: the English UD training set `en-ud-train.conllu` has a vocabulary with ~20k types. Of these ~10k occur just once. The dataset consists of 200k tokens though, so only 5% will end up belonging to this `<unk>` class.

* You will need dictionaries that map *words*, *tags*, and *labels* to indices: a `w2i`, `t2i`, and `l2i` dictionary. You also need the inverse dictionaries: `i2w`, `i2t`, `i2l`. (When your neural network predicts label 7 for some arc, you need to know which label that one was again to write out in the prediction!)
    * This is a good place to use a `defaultdict`! Have another look at the code in the PyTorch tutorial to see how you can do this (e.g. the code in `cbow.py` shows you).

* When you read in a `.conllu` file you will encounter some places where the line index is not an integer. E.g. it will say: `18.1	keep	keep	VERB	VB Mood=Imp|VerbForm=Fin	_	_	14:conj`. You might also find lines like: `1-2    vámonos   _ ...` - i.e. starting with integer-integer. You should remove these lines when reading in! These lines are not for our use.

* Write out the predictions of your trained parser in the CONNL-U format.
    * `Note` You only have to write out the parts of each line that we care about! That means the word *index*, the *word* itself, (optionally the *POS-tag*, if you saved this), the predicted incoming *arc*, and the predicted *label*. The rest of the information we discarded during read-in!
    * At the start of each sentence in a CONLL-U files there are a few lines that start with `#`. These lines contain meta-data about the sentence: an id, the raw text, etc. *You don’t have to write these out in your predictions!* (And you probably discarded this info in the first place). Since you probably have access to the words of your sentence you could write out the line: `# text = ...` where you put your words on the dots.

* When you write out your predictions the CONNL-U format it will be very easy to evaluate the performance. For this you can use the UD evaluation script that you can download [here](http://universaldependencies.org/conll17/evaluation.html).
    * The evaluation script is written in python and is super easy to use: if you saved your predicted CONLL-U file in `system_conllu_file` and the reference (gold-standard) CONLL-U file in `gold_conllu_file`, you can just call from the terminal: `python conll17_ud_eval.py -v gold_conllu_file system_conllu_file`, and the results will be printed out in the terminal.

----

## Graph algorithm

**Milestone:** Ideally, you have a working implementation of the *Chu-Liu-Edmonds' algorithm* for finding the maximum spanning-tree (MST).

But, we understand if you don’t yet. After all, this is **one of the most challenging parts of your project!** We don’t want your project to stall because you keep getting bugs in this part, so *in the unfortunate case where you just do not get it working, we can provide you with a completely error-free MST implementation.* This way, the success of your project will not hinge on this part.

### Tips

The additional readings of week three held another very good source the MST algorithm: the paper called [Non-projective Dependency Parsing using Spanning Tree Algorithms](http://www.aclweb.org/anthology/H05-1066). Figure 3 has a full pseudo-code.

----

## Neural Network

**Milestone:** Have a *minimal* neural network architecture set up. By this we mean the following:

* You have an **embedding layer for words**. For this you use the class `torch.nn.Embedding`, see [here](http://pytorch.org/docs/master/nn.html#embedding) for documentation. The indices in this layer correspond to the indices you assign to the words in your vocabulary. E.g. `<unk>` has index 0, `the` has index 1, etc. (Here you need your `w2i` dictionary!)

    * You can load in pre-trained word embeddings like GloVe into the `torch.nn.Embedding` class. [Here](https://discuss.pytorch.org/t/can-we-use-pre-trained-word-embeddings-for-weight-initialization-in-nn-embedding/1222) is how you can do that. (There is a slight caveat here. See the **note on training word embeddings**).

* You have an **embedding layer for POS-tags**. For this you use the same class `torch.nn.Embedding`. Now the indices of this layer correspond to the indices you assigned to your POS-tags. E.g. `DET` has index 0, `VERB` has index 1, etc.
    * Again you can load in your own POS-tag embeddings like above. Since this class is very small compared to your vocabulary, it probably won’t pay much to do this (in terms of computation time nor in performance).

* You **concatenate these word embeddings**: this takes `e_word` and `e_tag` and returns `e_word o e_tag` (where `o` indicates concatenation).

* You have an **LSTM** layer. This takes in the concatenated embeddings of all the word/POS-tag-pairs (`e_word o e_tag`) in the sentence, and returns their LSTM embeddings. For this you should use the class `torch.nn.LSTM` see [here](http://pytorch.org/docs/master/nn.html#embedding) for documentation.

**The above is the minimal setup we would like all the groups to have by the next week.**

Now, note that these first steps are the basis for both for the Kiperwasser & Goldberg model as well as the Dozat & Manning model. So from here on you can choose one of three directions:
1. Follow the Kiperwasser & Goldberg model for the final steps
2. Follow the Dozat & Manning model for the final steps
3. Follow a simpler architecture for the final steps. We have some suggestions for you here.

These final steps we will walk through together in the upcoming weeks. This includes showing you how to finish the above to take a full training step: using the LSTM embeddings to predict scores for each arcs; using these scores to predict arcs; predicting labels for these predicted arcs, calculating the loss of this combined prediction; using this loss to update the parameters.

### Tips

PyTorch automatically considers each instance of `torch.nn.Embedding` as a parameter that needs training. So even if you load in pretrained embeddings like described above, PyTorch will alter them during training. If you don’t want PyTorch to alter them during training (this will save your computer much computation time!) you must do the following.

Let `embedding` be the instance of the `torch.nn.Embedding` layer in which you uploaded the pre-trained embeddings. Let `model` be the instance of your model class of which `embedding` is a part. Then you should do:

```python
embedding.weight.requires_grad = False
parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.SGD(parameters, lr=learning_rate)
```
