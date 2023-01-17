#!/usr/bin/env python3
"""
Module contains function for computing the
n-gram BLEU score for a sentence generated by
a model, compared to reference sentences.
"""


import numpy as np


def gram(sen, start, n):
    """Gets an n-gram token of size n."""

    return [sen[start+i] for i in range(n)]


def n_grams(sen, n):
    """Gets an n-gram with tokens of size n."""
    return [gram(sen, i, n) for i in range(len(sen)-n+1)]


def bleu(references, sentence, n):
    """
    Calculates the n-gram BLEU score for a sentence
    generated by a model.
    Args:
        references: List of reference translations, where each
        reference translation is a list of the words in the translation.
        sentence: List containing the model proposed sentence.
        n: Size of the n-gram to use for evaluation.
    Return: n-gram BLEU score
    """

    ngrams = n_grams(sentence, n)
    tot, num_toks = 0, len(ngrams)

    while len(ngrams) > 0:
        tok = ngrams[0]
        cnt = ngrams.count(tok)
        for i in range(cnt):
            ngrams.pop(ngrams.index(tok))

        mx_ref = max([n_grams(ref, n).count(tok) for ref in references])

        tot += cnt if cnt <= mx_ref else mx_ref

    return tot / num_toks


def cumulative_bleu(references, sentence, n):
    """
    Calculates the cumulative n-gram BLEU score for a sentence
    generated by a model.
    Args:
        references: List of reference translations, where each
        reference translation is a list of the words in the translation.
        sentence: List containing the model proposed sentence.
        n: Size of the largest n-gram to use for evaluation.
    Return: cumulative n-gram BLEU score
    """

    min_ref, bp = min([len(ref) for ref in references]), 1

    if len(sentence) <= min_ref:
        bp = np.exp(1-min_ref/len(sentence))

    ps = [bleu(references, sentence, i) for i in range(1, n+1)]

    return bp * np.exp(np.log(ps).sum()/n)
