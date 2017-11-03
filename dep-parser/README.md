# Neural graph-based Dependency parser

In this project you will build a dependency parser from scratch!

## Introduction

In this project you will build a graph-based dependency parser in PyTorch.

A dependency tree can help to disambiguate the meaning of a sentence,
by uncovering the (kinds of) relations between its words:

- I saw [the girl] [with the telescope].
- I saw [the girl with the telecope].

Our dependency trees will also have labels, e.g. “I” is in a subject relation with “saw”.

The advantage of graph-based dependency parsers is that they can work well on languages with discontinuities,
such as Dutch and German, because we can extract non-projective dependency trees from them.

You will read relevant literature and re-implement the dependency parser of Dozat & Manning.
Next to English, you will choose one other language to investigate from the [Universal Dependencies project](http://universaldependencies.org/)

You will run a baseline dependency parser (out of the box) to get some scores that you will then try to beat with your own parser.

To extract dependency trees from your trained model, you will need an algorithm such as Eisner (for projective trees, suitable for languages such as English) and/or Chu-Liu-Edmonds (for non-projective trees, languages such as German) to find the minimum-spanning tree (MST) given the weights your model assigns between each pair of words.

## Required readings

1. J&M 3rd edition, [chapter Dependency parsing](https://web.stanford.edu/~jurafsky/slp3/)
  * Skip section 14.4 for now. In this section a so called *transition*-based parsing method is discussed; we will focus on the *graph*-based parsing method introduced in section 14.5
2. [Kiperwasser & Goldberg (2016)](https://aclweb.org/anthology/Q16-1023)
3. [Dozat & Manning (2017)](https://web.stanford.edu/~tdozat/files/TDozat-ICLR2017-Paper.pdf)
