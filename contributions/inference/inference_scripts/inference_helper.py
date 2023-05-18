import abc
import functools
import os
import torch
import torchtext
from ratsql.resources import corenlp
import matplotlib.pyplot as plt
import numpy as np

def generate_alignment_matrix_viz(m2c_align_mat, m2t_align_mat, nl_tokens, col_tokens, tbl_id):
    align_mat = torch.cat((m2c_align_mat, m2t_align_mat),1)
    n, m = len(nl_tokens), len(col_tokens)+1
    align_mat = align_mat[:n,:]
    align_mat = align_mat.cpu().detach().numpy()
    # Create a figure and axis object
    fig, ax = plt.subplots(figsize=(10, 15))

    # Set the colormap
    cmap = plt.cm.Reds

    # Create a heatmap
    heatmap = ax.pcolor(align_mat, cmap=cmap)

    # Set the ticks and labels
    # reverse the order of yaxis labels
    col_tokens_list = []
    for col in col_tokens:
        col_tokens_list.append(f"column:{col}")
    tbl_tokens_list = [f"table:{tbl_id}"]
    ytick_labels = nl_tokens
    xtick_labels = col_tokens_list + tbl_tokens_list
    yticks = np.arange(0.5, len(ytick_labels), 1)
    xticks = np.arange(0.5, len(xtick_labels), 1)
    
    ax.set_xticks(xticks, minor=False)
    ax.set_xticklabels(xtick_labels, minor=False, rotation=45, ha='right')
    ax.set_yticks(yticks, minor=False)
    ax.set_yticklabels(ytick_labels, minor=False)

    # Add the colorbar
    plt.colorbar(heatmap)

    # Set the title and axis labels
    plt.title('Alignment Matrix')
    # plt.xlabel('Columns')
    # plt.ylabel('Tokens')

    plt.savefig("sample.png")


class Embedder(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def tokenize(self, sentence):
        '''Given a string, return a list of tokens suitable for lookup.'''
        pass

    @abc.abstractmethod
    def untokenize(self, tokens):
        '''Undo tokenize.'''
        pass

    @abc.abstractmethod
    def lookup(self, token):
        '''Given a token, return a vector embedding if token is in vocabulary.

        If token is not in the vocabulary, then return None.'''
        pass

    @abc.abstractmethod
    def contains(self, token):
        pass

    @abc.abstractmethod
    def to(self, device):
        '''Transfer the pretrained embeddings to the given device.'''
        pass

class GloVe(Embedder):

    def __init__(self, folder_path, kind, lemmatize=False):
        cache = os.path.join(folder_path, '.vector_cache')
        self.glove = torchtext.vocab.GloVe(name=kind, cache=cache)
        self.dim = self.glove.dim
        self.vectors = self.glove.vectors
        self.lemmatize = lemmatize
        self.corenlp_annotators = ['tokenize', 'ssplit']
        if lemmatize:
            self.corenlp_annotators.append('lemma')

    @functools.lru_cache(maxsize=1024)
    def tokenize(self, text):
        ann = corenlp.annotate(text, self.corenlp_annotators)
        if self.lemmatize:
            return [tok.lemma.lower() for sent in ann.sentence for tok in sent.token]
        else:
            return [tok.word.lower() for sent in ann.sentence for tok in sent.token]
    
    @functools.lru_cache(maxsize=1024)
    def tokenize_for_copying(self, text):
        ann = corenlp.annotate(text, self.corenlp_annotators)
        text_for_copying = [tok.originalText.lower() for sent in ann.sentence for tok in sent.token]
        if self.lemmatize:
            text = [tok.lemma.lower() for sent in ann.sentence for tok in sent.token]
        else:
            text = [tok.word.lower() for sent in ann.sentence for tok in sent.token]
        return text, text_for_copying

    def untokenize(self, tokens):
        return ' '.join(tokens)

    def lookup(self, token):
        i = self.glove.stoi.get(token)
        if i is None:
            return None
        return self.vectors[i]

    def contains(self, token):
        return token in self.glove.stoi

    def to(self, device):
        self.vectors = self.vectors.to(device)