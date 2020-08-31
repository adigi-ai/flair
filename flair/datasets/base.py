import logging
from abc import abstractmethod
from pathlib import Path
from typing import List, Union, Callable

import torch.utils.data.dataloader
from torch.utils.data.dataset import Subset, ConcatDataset

from flair.data import (
    Sentence,
    Token,
    FlairDataset,
    space_tokenizer
)


log = logging.getLogger("flair")


class DataLoader(torch.utils.data.dataloader.DataLoader):
    def __init__(
            self,
            dataset,
            batch_size=1,
            shuffle=False,
            sampler=None,
            batch_sampler=None,
            num_workers=8,
            drop_last=False,
            timeout=0,
            worker_init_fn=None,
    ):

        # in certain cases, multi-CPU data loading makes no sense and slows
        # everything down. For this reason, we detect if a dataset is in-memory:
        # if so, num_workers is set to 0 for faster processing
        flair_dataset = dataset
        while True:
            if type(flair_dataset) is Subset:
                flair_dataset = flair_dataset.dataset
            elif type(flair_dataset) is ConcatDataset:
                flair_dataset = flair_dataset.datasets[0]
            else:
                break

        if type(flair_dataset) is list:
            num_workers = 0
        elif isinstance(flair_dataset, FlairDataset) and flair_dataset.is_in_memory():
            num_workers = 0

        super(DataLoader, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=list,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )


class SentenceDataset(FlairDataset):
    """
    A simple Dataset object to wrap a List of Sentence
    """

    def __init__(self, sentences: Union[Sentence, List[Sentence]]):
        """
        Instantiate SentenceDataset
        :param sentences: Sentence or List of Sentence that make up SentenceDataset
        """
        # cast to list if necessary
        if type(sentences) == Sentence:
            sentences = [sentences]
        self.sentences = sentences

    @abstractmethod
    def is_in_memory(self) -> bool:
        return True

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index: int = 0) -> Sentence:
        return self.sentences[index]


class StringDataset(FlairDataset):
    """
    A Dataset taking string as input and returning Sentence during iteration
    """

    def __init__(
            self,
            texts: Union[str, List[str]],
            use_tokenizer: Union[bool, Callable[[str], List[Token]]] = space_tokenizer,
    ):
        """
        Instantiate StringDataset
        :param texts: a string or List of string that make up StringDataset
        :param use_tokenizer: a custom tokenizer (default is space based tokenizer,
        more advanced options are segtok_tokenizer to use segtok or build_spacy_tokenizer to use Spacy library
        if available). Check the code of space_tokenizer to implement your own (if you need it).
        If instead of providing a function, this parameter is just set to True, segtok will be used.
        """
        # cast to list if necessary
        if type(texts) == Sentence:
            texts = [texts]
        self.texts = texts
        self.use_tokenizer = use_tokenizer

    @abstractmethod
    def is_in_memory(self) -> bool:
        return True

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index: int = 0) -> Sentence:
        text = self.texts[index]
        return Sentence(text, use_tokenizer=self.use_tokenizer)
