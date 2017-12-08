# Neural graph-based dependency parser

In this project you will build a dependency parser from scratch!

`New!` Check the [MST tutorial](mst/) with code that you may use for your project.

Read the [milestones](milestone) for more tips.

## Introduction

In this project you will build a **graph-based dependency parser** that is trained using **neurals networks** with the help of **PyTorch**.

Concretely, you will:

* Read the relevant literature on dependency grammars, graph algorithms, and neural networks.
* Use this to re-implement the model described in [Kiperwasser & Goldberg (2016)](https://aclweb.org/anthology/Q16-1023) or to re-implement the extension of this model described in [Dozat & Manning (2017)](https://arxiv.org/abs/1611.01734). In any case, you should read Kipperwasser & Goldberg first.
* Train this model on annotated data from the [Universal Dependencies project](http://universaldependencies.org/). Next to English, you will choose one other language to investigate. Ideally you choose a language that you are familiar with, so that you can interpret the performance of you model!
* Use the trained model to parse sentences in a test-set, and evaluate how well it performs.
* `Optional` Run a baseline dependency parser (that we provide) to get some scores. See if you can beat them with your own parser!
* Write a final report on the results of your experiments.

At the end of the project you will have a fully working parser! If time permits, you can do a number of interesting things with it:

* Test its performance on very hard sentences. Look [here](https://en.wikipedia.org/wiki/List_of_linguistic_example_sentences#cite_note-1) for inspiration!
* Investigate the type of grammatical errors your parser makes, and try to understand what causes them.
* Inspect the parameters of the trained neural network to see what it has learned.
* Or you could even come up with an improvement to the model!

Read on for more details on the project.

Under the section **sources** I collect sources that you might find useful. Note that you are by no means required to read them all! Browse them and see for yourself which sources you find useful - and until you found what you needed.



## Dependency grammar

**Dependency grammar** is a grammatical formalism that is based on word-word relations. This means that the grammatical structure of a sentence is described solely in terms of the **words** in a sentence and the **syntactic relations** between the words.

Here is an example of a sentence annotated with dependency relations:

![example](dependency-example.png)

(We will refer to the above interchangeably as: a **dependency parse**; a **dependency graph**; and a **dependency tree**.)

**Dependency parsing** is the task of taking a bare (unannotated) sentence like *I prefer the morning flight through Denver* and then assigning a syntactic structure to it. If our parser is good, it will assign (roughly) the parse in the image above.

### Sources

* Jurafsky and Manning, chapter [Dependency parsing](Jurafsky&ManningCh14.pdf) (3rd edition) is your main source of information on dependency grammars and dependency parsing.
* If you want additional information you can look at the slides from [this course](http://cl.indiana.edu/~md7/nasslli10/). In particular the slides on [Dependency Grammar](http://cl.indiana.edu/~md7/nasslli10/01/01-grammar.pdf) and [Graph-Based Dependency Parsing](http://cl.indiana.edu/~md7/nasslli10/04/graphbased.pdf) .
* We use data from the [Universal Dependencies project](http://universaldependencies.org/). The [annotation guidelines](http://universaldependencies.org/guidelines.html) contains the answers all your data-format related questions.

## Graph algorithms

This project is about **graph-based** dependency parsing.

Concretely, this means we you will use a graph algorithm like [Chu-Liu-Edmonds' algorithm](https://en.wikipedia.org/wiki/Edmonds%27_algorithm) or [Eisner's algorithm](http://curtis.ml.cmu.edu/w/courses/index.php/Eisner_algorithm) to parse a sentence. These algorithms are used to find the so called **minimum-spanning-tree** [(MST)](https://en.wikipedia.org/wiki/Minimum_spanning_tree) in a graph. They are general graph algorithms used for all kinds of discrete optimisation tasks. The way **we** will use these algorithms is as follows.

Let's say we have some sentence. Our model assigns **weights** to **all possible arcs** between the words in that sentence (more below about how we get these weights). This gives us a **complete graph** on all the words in the sentence, with **weighted arcs** between the words. Then, we use one of the above algorithms to obtain the minimum-spanning tree in this complete graph. What we obtain is the predicted **dependency tree** for the sentence under consideration.

The following image gives a good visual summary of the process for the sentence *Kasey hugged Kim*:

![hug-MST](kasey-hugged-kim-MST.png)

The dotted lines show the complete graph. Each of these is assigned a weight by our model. We then run the MST algorithm. The solid lines show the obtained minimum-spanning-tree. This gives use then the dependency parse:

![hug](kasey-hugged-kim.png)

Note: the **labels** on the arcs are **not** obtained using this algorithm. They are predicted afterwards. (We will discuss this later)

### Sources

* A good source for the MST algorithm is the paper called [Non-projective Dependency Parsing using Spanning Tree Algorithms](http://www.aclweb.org/anthology/H05-1066). Figure 3 has a full pseudo-code.
* The notebook [mst.ipynb](mst/mst.ipynb) has a reference implementation of the MST algorithm that you can use to test your implementation, or to use in case you cannot get your implementation error-free.
* There is a python package for graphs called [NetworkX](http://networkx.github.io/) that has an easy to use data-structure for representing [graphs](https://networkx.github.io/documentation/stable/reference/classes/index.html)), and implementation of [Edmond's algorithm](https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.tree.branchings.Edmonds.html?highlight=edmonds) that you can use to check the correctness of your own implementation. Lastly, it let's you [draw](https://networkx.github.io/documentation/stable/reference/generated/networkx.drawing.nx_pylab.draw.html?highlight=draw#networkx.drawing.nx_pylab.draw) graphs, or [save](https://networkx.github.io/documentation/stable/reference/readwrite/graphml.html?highlight=xml) them as xml file so that you can draw them with other graph-drawing packages.
* See the notebook [graphs.ipynb](notebooks/graphs.ipynb) for a small demo on NetworkX.

## Neural networks

For the method above to work well, we need to assign **weights** to all the possible edges. These weights are crucial for obtaining good parses; they essentially control which tree we obtain! But how do get them? For this we will use a **neural network**.

### Embeddings

Given a sentence of length `n`, we take the `n` **words** with their **POS-tags** to their vector-representations (embeddings). Concatenate them to give `n` word/POS-tag embeddings.

`[Under development]`

### LSTM

Pass the concatenated word/POS-tag embeddings through an LSTM layer to get a new set of `n` vectors.

`[Under development]`

### Scoring

Use the set of `n` LSTM-output vectors to get scores for all `n^2` possible arcs `w_i -> w_j`.

This gives us an `n x n` score matrix `S` such that `S[i,j] = score(w_i -> w_j)`, where the score is **any real number**/

`[Under development]`

### Training objective

From the matrix `S` we can obtain the matrix `A` such that `A[i,j] = p(w_i -> w_j)`. The interpretation of `A` is that each column specifies a probability distribution `p(i -> j)`, for `i=0,1,..,n`, such that `sum_i p(i -> j) = 1`. Hence we see this as a kind of peculiar classification problem: each of the `n` words `w_i` is classified into one of `n` classes, the word `w_j` that is its head.To turn `S` into `A` we need to turn the **columns** into probability distributions. For this you use the [**softmax**](https://en.wikipedia.org/wiki/Softmax_function) function.

The **training objective** for the neural network is to minimise the [**cross-entropy loss**](https://en.wikipedia.org/wiki/Cross_entropy) between the **column** `a_i` of the **predicted** adjacency matrix `A` and the **columns** `g_i` of the **gold** adjacency matrix `G`.

Here's a gif of what that looks like. What you see below is the adjacency matrix `A` as it develops during training. The first and the last frame show the gold 0-1 adjacency matrix `G`.

![adjacency-gif](adjacency.gif)


#### Pytorch

In terms of **PyTorch** implementation of softmax and cross-entropy loss there are some particulars:
* The function `torch.nn.CrossEntropyLoss` ([link](http://pytorch.org/docs/master/nn.html#torch.nn.CrossEntropyLoss)) accepts a tensor of shape `[minibatch, num_classes]`, with **logits**: *un-normalized* log-probabilities. In our case: this is the **score-matrix before applying the softmax**.
    * You can process the whole score matrix **in one go**: minibatch of size `n` that is classified into one of `n` classes. Take another look at the matrix `A` above to see this.
    * Make sure you get the dimensions right! For example, the matrix `A` above is actually in the format `[num_classes, minibatch]`, (you see why?).

The documentation for the function `CrossEntropyLoss` mentions: *This criterion combines* `LogSoftMax` *and* `NLLLoss` *in one single class.* What does that mean?

* The function `torch.nn.LogSoftmax` ([link](http://pytorch.org/docs/master/nn.html?highlight=softmax#torch.nn.LogSoftmax)) computes the log-softmax of the input tensor. This gives you **log-probabilities**. You can specify the dimension to normalize.

* The function `torch.nn.NLLLoss` ([link](http://pytorch.org/docs/master/nn.html#torch.nn.NLLLoss)) accepts a tensor of shape `[minibatch, num_classes]` with **log-probabilities**, and computes `loss(x, class) = -x[class]`. This is the cross-entropy between `x` and a one hot distribution for the class label!

Hence, there are two 'different' ways to the same loss: If you first compute the `LogSoftmax` of the score matrix `S` and then use `NLLloss` you get the same loss as when you give the score matrix `S` to `CrossEntropyLoss` directly. **It is up to you which you choose.**

### Sources

In this section we collect sources that we think are useful for understanding the neural network methods used in this projects.

* **Word embeddings**
  * [Chapter 15](https://web.stanford.edu/~jurafsky/slp3/15.pdf) and especially [chapter 16](https://web.stanford.edu/~jurafsky/slp3/16.pdf) of Jurafsky and Martin (3rd edition) is a good reference on the idea of using vector representations for words.
  * Word embeddings are often visualised with **t-SNE**. Learn more about that on the [author's webpage](https://lvdmaaten.github.io/tsne/) or from the publication [How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/) on Distill.
  * **The** [**notebook on word embeddings**](notebooks/word-embeddings.ipynb) that we looked at in class.

* **Sources for word embeddings**:
  * [Gensim](https://radimrehurek.com/gensim/models/word2vec.html) has an implementation of [Word2Vec](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/) that you can use to make your own word embeddings.
  * [GloVe](https://nlp.stanford.edu/projects/glove/) are perhaps the best vector representations for words available. You can download them in all kinds of dimensions.
  * [Dependency-base word embeddings](https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/) (see [here](http://www.aclweb.org/anthology/P14-2050) for the paper) are like Word2Vec, but with the big difference that context here is not defined as "nearby in linear distance" but as "nearby in a syntactic tree". Simple example: in the parse for *I prefer the morning flight through Denver* (see above), the word *flight* has a linear distance of 3 from the word *prefer*, but a syntactic distance of just 1. This gives embeddings that are more effective at capturing syntactic information than regular the type of word embeddings described above. Looking at the effect this has could be an interesting experiment to perform with your finished parser! **Note:** the only downside is that these embeddings are only available for download in dimension 300 (which is rather big for our purpose!).

* **[Recurrent Neural Networks](https://en.wikipedia.org/wiki/Recurrent_neural_network)** (RNNs) and in particular **[LSTMs](https://en.wikipedia.org/wiki/Long_short-term_memory)**
  * LSTM stands for *Long short-term memory*. For clarity: an LSTM is a special type of RNN.
  * [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) by Andrej Karpathy.
  * [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) on Christopher Olah's blog.
  * In class we looked at these slides from the Oxford course on Deep Learning for NLP: [Lecture 3](https://github.com/oxford-cs-deepnlp-2017/lectures/blob/master/Lecture%203%20-%20Language%20Modelling%20and%20RNNs%20Part%201.pdf) and [lecture 4](https://github.com/oxford-cs-deepnlp-2017/lectures/blob/master/Lecture%204%20-%20Language%20Modelling%20and%20RNNs%20Part%202.pdf).

* **Pytorch**
  * This [PyTorch tutorial](http://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html) has a simple implementation of an RNN.
  * Another [PyTorch tutorial](http://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html?highlight=lstm) on how to use and LSTM for POS-tagging (without an HMM, Viterbi, or the forward-backward probabilities!).
  * The PyTorch documentation on [recurrent layers](http://pytorch.org/docs/master/nn.html#recurrent-layers) in the NN module has out of the box implementations of RNNs and LSTMs.
  * The PyTorch documentation on [sparse layers](http://pytorch.org/docs/master/nn.html#embedding) in the NN module contains the class `Embedding`. This implements the embedding matrix that I discussed in class in an extremely easy-to-use way.

## Required readings

The following sources are the theoretical backbone of the project:

1. J&M 3rd edition, chapter [Dependency parsing](Jurafsky&ManningCh14.pdf). Skip section 14.4 for now. In this section a so called *transition*-based parsing method is discussed; we will focus on the *graph*-based parsing method introduced in section 14.5.
2. [Kiperwasser & Goldberg (2016)](https://aclweb.org/anthology/Q16-1023)
3. [Dozat & Manning (2017)](https://arxiv.org/abs/1611.01734) (also see their [poster](TDozat-ICLR2017-Poster.pdf) and [slides](TDozat-CoNLL2017-Presentation.pdf)!)

We advice you to read them in this order, especially the last two papers: Dozat & Manning (2017) is an extension of the model of Kiperwasser & Goldberg (2016), and the former presupposes a lot of knowledge of the latter.

Note that the Kiperwasser & Goldberg paper is rather dense, but very complete. It contains condensed but very good explanations of all the techniques and steps taken in the implementation. So study this paper carefully! Then the extension that Dozat & Manning propose will make a lot more sense.
