from typing import List

import flair
from flair.data import Sentence


def store_embeddings(sentences: List[Sentence], storage_mode: str):

    # if memory mode option 'none' delete everything
    if storage_mode == "none":
        for sentence in sentences:
            sentence.clear_embeddings()

    # else delete only dynamic embeddings (otherwise autograd will keep everything in memory)
    else:
        # find out which ones are dynamic embeddings
        delete_keys = []
        if type(sentences[0]) == Sentence:
            for name, vector in sentences[0][0]._embeddings.items():
                if sentences[0][0]._embeddings[name].requires_grad:
                    delete_keys.append(name)

        # find out which ones are dynamic embeddings
        for sentence in sentences:
            sentence.clear_embeddings(delete_keys)

    # memory management - option 1: send everything to CPU (pin to memory if we train on GPU)
    if storage_mode == "cpu":
        pin_memory = False if str(flair.device) == "cpu" else True
        for sentence in sentences:
            sentence.to("cpu", pin_memory=pin_memory)

    # record current embedding storage mode to allow optimization (for instance in FlairEmbeddings class)
    flair.embedding_storage_mode = storage_mode
