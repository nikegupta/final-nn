# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike
import random
def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    #seed random
    random.seed(1)

    #count the number of positive and negative
    num_positive = 0
    num_negative = 0
    for i in range(len(labels)):
        if labels[i] == 0:
            num_negative += 1
        elif labels[i] == 1:
            num_positive += 1
        else:
            raise ValueError('Illegal label detected')

    #take the min number, will take that number of positives and negatives 
    min_number = min([num_positive,num_negative])

    #find the idx of positive and negative samples
    labels_np = np.array(labels)
    pos_idx = np.where(labels_np == 1)[0]
    neg_idx = np.where(labels_np == 0)[0]

    #randomly sample min_number of them
    pos_idx_sampled = np.random.choice(pos_idx,min_number,replace=False)
    neg_idx_sampled = np.random.choice(neg_idx,min_number,replace=False)

    #add the sampled indexes
    pre_sampled_seqs = []
    pre_sampled_labels = []
    for i in pos_idx_sampled:
        pre_sampled_seqs.append(seqs[i])
        pre_sampled_labels.append(labels[i])
    for i in neg_idx_sampled:
        pre_sampled_seqs.append(seqs[i])
        pre_sampled_labels.append(labels[i])

    #instead of having pos:neg, mix them together
    sampled_seqs = []
    sampled_labels = []
    random_idx = list(range(len(pre_sampled_seqs)))
    random.shuffle(random_idx)
    for i in random_idx:
        sampled_seqs.append(pre_sampled_seqs[i])
        sampled_labels.append(pre_sampled_labels[i])

    return sampled_seqs, sampled_labels

def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    encodings = []
    for i in range(len(seq_arr)):
        encoding = []
        
        for j in range(len(seq_arr[i])):
            if seq_arr[i][j] == 'A':
                encoding.append(1)
                encoding.append(0)
                encoding.append(0)
                encoding.append(0)
            elif seq_arr[i][j] == 'T':
                encoding.append(0)
                encoding.append(1)
                encoding.append(0)
                encoding.append(0)
            elif seq_arr[i][j] == 'C':
                encoding.append(0)
                encoding.append(0)
                encoding.append(1)
                encoding.append(0)
            elif seq_arr[i][j] == 'G':
                encoding.append(0)
                encoding.append(0)
                encoding.append(0)
                encoding.append(1)
            else:
                raise ValueError('Non ATGC character found')

        encodings.append(encoding)

    return encodings