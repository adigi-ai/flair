from abc import abstractmethod
from pathlib import Path
from typing import List, Union

import torch

import flair
import logging

from flair.data import Sentence, Token, Dictionary
from flair.embeddings.base import Embeddings
from flair.file_utils import cached_path, open_inside_zip

log = logging.getLogger("flair")


class TokenEmbeddings(Embeddings):
    """Abstract base class for all token-level embeddings. Ever new type of word embedding must implement these methods."""

    @property
    @abstractmethod
    def embedding_length(self) -> int:
        """Returns the length of the embedding vector."""
        pass

    @property
    def embedding_type(self) -> str:
        return "word-level"


class StackedEmbeddings(TokenEmbeddings):
    """A stack of embeddings, used if you need to combine several different embedding types."""

    def __init__(self, embeddings: List[TokenEmbeddings]):
        """The constructor takes a list of embeddings to be combined."""
        super().__init__()

        self.embeddings = embeddings

        # IMPORTANT: add embeddings as torch modules
        for i, embedding in enumerate(embeddings):
            embedding.name = f"{str(i)}-{embedding.name}"
            self.add_module(f"list_embedding_{str(i)}", embedding)

        self.name: str = "Stack"
        self.static_embeddings: bool = True

        self.__embedding_type: str = embeddings[0].embedding_type

        self.__embedding_length: int = 0
        for embedding in embeddings:
            self.__embedding_length += embedding.embedding_length

    def embed(
        self, sentences: Union[Sentence, List[Sentence]], static_embeddings: bool = True
    ):
        # if only one sentence is passed, convert to list of sentence
        if type(sentences) is Sentence:
            sentences = [sentences]

        for embedding in self.embeddings:
            embedding.embed(sentences)

    @property
    def embedding_type(self) -> str:
        return self.__embedding_type

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        for embedding in self.embeddings:
            embedding._add_embeddings_internal(sentences)

        return sentences

    def __str__(self):
        return f'StackedEmbeddings [{",".join([str(e) for e in self.embeddings])}]'

class CharacterEmbeddings(TokenEmbeddings):
    """Character embeddings of words, as proposed in Lample et al., 2016."""

    def __init__(
        self,
        path_to_char_dict: str = None,
        char_embedding_dim: int = 25,
        hidden_size_char: int = 25,
    ):
        """Uses the default character dictionary if none provided."""

        super().__init__()
        self.name = "Char"
        self.static_embeddings = False

        # use list of common characters if none provided
        if path_to_char_dict is None:
            self.char_dictionary: Dictionary = Dictionary.load("common-chars")
        else:
            self.char_dictionary: Dictionary = Dictionary.load_from_file(
                path_to_char_dict
            )

        self.char_embedding_dim: int = char_embedding_dim
        self.hidden_size_char: int = hidden_size_char
        self.char_embedding = torch.nn.Embedding(
            len(self.char_dictionary.item2idx), self.char_embedding_dim
        )
        self.char_rnn = torch.nn.LSTM(
            self.char_embedding_dim,
            self.hidden_size_char,
            num_layers=1,
            bidirectional=True,
        )

        self.__embedding_length = self.hidden_size_char * 2

        self.to(flair.device)

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]):

        for sentence in sentences:

            tokens_char_indices = []

            # translate words in sentence into ints using dictionary
            for token in sentence.tokens:
                char_indices = [
                    self.char_dictionary.get_idx_for_item(char) for char in token.text
                ]
                tokens_char_indices.append(char_indices)

            # sort words by length, for batching and masking
            tokens_sorted_by_length = sorted(
                tokens_char_indices, key=lambda p: len(p), reverse=True
            )
            d = {}
            for i, ci in enumerate(tokens_char_indices):
                for j, cj in enumerate(tokens_sorted_by_length):
                    if ci == cj:
                        d[j] = i
                        continue
            chars2_length = [len(c) for c in tokens_sorted_by_length]
            longest_token_in_sentence = max(chars2_length)
            tokens_mask = torch.zeros(
                (len(tokens_sorted_by_length), longest_token_in_sentence),
                dtype=torch.long,
                device=flair.device,
            )

            for i, c in enumerate(tokens_sorted_by_length):
                tokens_mask[i, : chars2_length[i]] = torch.tensor(
                    c, dtype=torch.long, device=flair.device
                )

            # chars for rnn processing
            chars = tokens_mask

            character_embeddings = self.char_embedding(chars).transpose(0, 1)

            packed = torch.nn.utils.rnn.pack_padded_sequence(
                character_embeddings, chars2_length
            )

            lstm_out, self.hidden = self.char_rnn(packed)

            outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out)
            outputs = outputs.transpose(0, 1)
            chars_embeds_temp = torch.zeros(
                (outputs.size(0), outputs.size(2)),
                dtype=torch.float,
                device=flair.device,
            )
            for i, index in enumerate(output_lengths):
                chars_embeds_temp[i] = outputs[i, index - 1]
            character_embeddings = chars_embeds_temp.clone()
            for i in range(character_embeddings.size(0)):
                character_embeddings[d[i]] = chars_embeds_temp[i]

            for token_number, token in enumerate(sentence.tokens):
                token.set_embedding(self.name, character_embeddings[token_number])

    def __str__(self):
        return self.name


class FlairEmbeddings(TokenEmbeddings):
    """Contextual string embeddings of words, as proposed in Akbik et al., 2018."""

    def __init__(self,
                 model,
                 fine_tune: bool = False,
                 chars_per_chunk: int = 512,
                 with_whitespace: bool = True,
                 tokenized_lm: bool = True,
                 ):
        """
        initializes contextual string embeddings using a character-level language model.
        :param model: model string, one of 'news-forward', 'news-backward', 'news-forward-fast', 'news-backward-fast',
                'mix-forward', 'mix-backward', 'german-forward', 'german-backward', 'polish-backward', 'polish-forward',
                etc (see https://github.com/flairNLP/flair/blob/master/resources/docs/embeddings/FLAIR_EMBEDDINGS.md)
                depending on which character language model is desired.
        :param fine_tune: if set to True, the gradient will propagate into the language model. This dramatically slows
                down training and often leads to overfitting, so use with caution.
        :param chars_per_chunk: max number of chars per rnn pass to control speed/memory tradeoff. Higher means faster
                but requires more memory. Lower means slower but less memory.
        :param with_whitespace: If True, use hidden state after whitespace after word. If False, use hidden
                 state at last character of word.
        :param tokenized_lm: Whether this lm is tokenized. Default is True, but for LMs trained over unprocessed text
                False might be better.
        """
        super().__init__()

        cache_dir = Path("embeddings")

        aws_path: str = "https://s3.eu-central-1.amazonaws.com/alan-nlp/resources"
        hu_path: str = "https://flair.informatik.hu-berlin.de/resources"
        clef_hipe_path: str = "https://files.ifi.uzh.ch/cl/siclemat/impresso/clef-hipe-2020/flair"

        self.PRETRAINED_MODEL_ARCHIVE_MAP = {
            # multilingual models
            "multi-forward": f"{aws_path}/embeddings-v0.4.3/lm-jw300-forward-v0.1.pt",
            "multi-backward": f"{aws_path}/embeddings-v0.4.3/lm-jw300-backward-v0.1.pt",
            "multi-v0-forward": f"{aws_path}/embeddings-v0.4/lm-multi-forward-v0.1.pt",
            "multi-v0-backward": f"{aws_path}/embeddings-v0.4/lm-multi-backward-v0.1.pt",
            "multi-v0-forward-fast": f"{aws_path}/embeddings-v0.4/lm-multi-forward-fast-v0.1.pt",
            "multi-v0-backward-fast": f"{aws_path}/embeddings-v0.4/lm-multi-backward-fast-v0.1.pt",
            # English models
            "en-forward": f"{aws_path}/embeddings-v0.4.1/big-news-forward--h2048-l1-d0.05-lr30-0.25-20/news-forward-0.4.1.pt",
            "en-backward": f"{aws_path}/embeddings-v0.4.1/big-news-backward--h2048-l1-d0.05-lr30-0.25-20/news-backward-0.4.1.pt",
            "en-forward-fast": f"{aws_path}/embeddings/lm-news-english-forward-1024-v0.2rc.pt",
            "en-backward-fast": f"{aws_path}/embeddings/lm-news-english-backward-1024-v0.2rc.pt",
            "news-forward": f"{aws_path}/embeddings-v0.4.1/big-news-forward--h2048-l1-d0.05-lr30-0.25-20/news-forward-0.4.1.pt",
            "news-backward": f"{aws_path}/embeddings-v0.4.1/big-news-backward--h2048-l1-d0.05-lr30-0.25-20/news-backward-0.4.1.pt",
            "news-forward-fast": f"{aws_path}/embeddings/lm-news-english-forward-1024-v0.2rc.pt",
            "news-backward-fast": f"{aws_path}/embeddings/lm-news-english-backward-1024-v0.2rc.pt",
            "mix-forward": f"{aws_path}/embeddings/lm-mix-english-forward-v0.2rc.pt",
            "mix-backward": f"{aws_path}/embeddings/lm-mix-english-backward-v0.2rc.pt",
            # Arabic
            "ar-forward": f"{aws_path}/embeddings-stefan-it/lm-ar-opus-large-forward-v0.1.pt",
            "ar-backward": f"{aws_path}/embeddings-stefan-it/lm-ar-opus-large-backward-v0.1.pt",
            # Bulgarian
            "bg-forward-fast": f"{aws_path}/embeddings-v0.3/lm-bg-small-forward-v0.1.pt",
            "bg-backward-fast": f"{aws_path}/embeddings-v0.3/lm-bg-small-backward-v0.1.pt",
            "bg-forward": f"{aws_path}/embeddings-stefan-it/lm-bg-opus-large-forward-v0.1.pt",
            "bg-backward": f"{aws_path}/embeddings-stefan-it/lm-bg-opus-large-backward-v0.1.pt",
            # Czech
            "cs-forward": f"{aws_path}/embeddings-stefan-it/lm-cs-opus-large-forward-v0.1.pt",
            "cs-backward": f"{aws_path}/embeddings-stefan-it/lm-cs-opus-large-backward-v0.1.pt",
            "cs-v0-forward": f"{aws_path}/embeddings-v0.4/lm-cs-large-forward-v0.1.pt",
            "cs-v0-backward": f"{aws_path}/embeddings-v0.4/lm-cs-large-backward-v0.1.pt",
            # Danish
            "da-forward": f"{aws_path}/embeddings-stefan-it/lm-da-opus-large-forward-v0.1.pt",
            "da-backward": f"{aws_path}/embeddings-stefan-it/lm-da-opus-large-backward-v0.1.pt",
            # German
            "de-forward": f"{aws_path}/embeddings/lm-mix-german-forward-v0.2rc.pt",
            "de-backward": f"{aws_path}/embeddings/lm-mix-german-backward-v0.2rc.pt",
            "de-historic-ha-forward": f"{aws_path}/embeddings-stefan-it/lm-historic-hamburger-anzeiger-forward-v0.1.pt",
            "de-historic-ha-backward": f"{aws_path}/embeddings-stefan-it/lm-historic-hamburger-anzeiger-backward-v0.1.pt",
            "de-historic-wz-forward": f"{aws_path}/embeddings-stefan-it/lm-historic-wiener-zeitung-forward-v0.1.pt",
            "de-historic-wz-backward": f"{aws_path}/embeddings-stefan-it/lm-historic-wiener-zeitung-backward-v0.1.pt",
            "de-historic-rw-forward": f"{hu_path}/embeddings/redewiedergabe_lm_forward.pt",
            "de-historic-rw-backward": f"{hu_path}/embeddings/redewiedergabe_lm_backward.pt",
            # Spanish
            "es-forward": f"{aws_path}/embeddings-v0.4/language_model_es_forward_long/lm-es-forward.pt",
            "es-backward": f"{aws_path}/embeddings-v0.4/language_model_es_backward_long/lm-es-backward.pt",
            "es-forward-fast": f"{aws_path}/embeddings-v0.4/language_model_es_forward/lm-es-forward-fast.pt",
            "es-backward-fast": f"{aws_path}/embeddings-v0.4/language_model_es_backward/lm-es-backward-fast.pt",
            # Basque
            "eu-forward": f"{aws_path}/embeddings-stefan-it/lm-eu-opus-large-forward-v0.2.pt",
            "eu-backward": f"{aws_path}/embeddings-stefan-it/lm-eu-opus-large-backward-v0.2.pt",
            "eu-v1-forward": f"{aws_path}/embeddings-stefan-it/lm-eu-opus-large-forward-v0.1.pt",
            "eu-v1-backward": f"{aws_path}/embeddings-stefan-it/lm-eu-opus-large-backward-v0.1.pt",
            "eu-v0-forward": f"{aws_path}/embeddings-v0.4/lm-eu-large-forward-v0.1.pt",
            "eu-v0-backward": f"{aws_path}/embeddings-v0.4/lm-eu-large-backward-v0.1.pt",
            # Persian
            "fa-forward": f"{aws_path}/embeddings-stefan-it/lm-fa-opus-large-forward-v0.1.pt",
            "fa-backward": f"{aws_path}/embeddings-stefan-it/lm-fa-opus-large-backward-v0.1.pt",
            # Finnish
            "fi-forward": f"{aws_path}/embeddings-stefan-it/lm-fi-opus-large-forward-v0.1.pt",
            "fi-backward": f"{aws_path}/embeddings-stefan-it/lm-fi-opus-large-backward-v0.1.pt",
            # French
            "fr-forward": f"{aws_path}/embeddings/lm-fr-charlm-forward.pt",
            "fr-backward": f"{aws_path}/embeddings/lm-fr-charlm-backward.pt",
            # Hebrew
            "he-forward": f"{aws_path}/embeddings-stefan-it/lm-he-opus-large-forward-v0.1.pt",
            "he-backward": f"{aws_path}/embeddings-stefan-it/lm-he-opus-large-backward-v0.1.pt",
            # Hindi
            "hi-forward": f"{aws_path}/embeddings-stefan-it/lm-hi-opus-large-forward-v0.1.pt",
            "hi-backward": f"{aws_path}/embeddings-stefan-it/lm-hi-opus-large-backward-v0.1.pt",
            # Croatian
            "hr-forward": f"{aws_path}/embeddings-stefan-it/lm-hr-opus-large-forward-v0.1.pt",
            "hr-backward": f"{aws_path}/embeddings-stefan-it/lm-hr-opus-large-backward-v0.1.pt",
            # Indonesian
            "id-forward": f"{aws_path}/embeddings-stefan-it/lm-id-opus-large-forward-v0.1.pt",
            "id-backward": f"{aws_path}/embeddings-stefan-it/lm-id-opus-large-backward-v0.1.pt",
            # Italian
            "it-forward": f"{aws_path}/embeddings-stefan-it/lm-it-opus-large-forward-v0.1.pt",
            "it-backward": f"{aws_path}/embeddings-stefan-it/lm-it-opus-large-backward-v0.1.pt",
            # Japanese
            "ja-forward": f"{aws_path}/embeddings-v0.4.1/lm__char-forward__ja-wikipedia-3GB/japanese-forward.pt",
            "ja-backward": f"{aws_path}/embeddings-v0.4.1/lm__char-backward__ja-wikipedia-3GB/japanese-backward.pt",
            # Malayalam
            "ml-forward": f"https://raw.githubusercontent.com/qburst/models-repository/master/FlairMalayalamModels/ml-forward.pt",
            "ml-backward": f"https://raw.githubusercontent.com/qburst/models-repository/master/FlairMalayalamModels/ml-backward.pt",
            # Dutch
            "nl-forward": f"{aws_path}/embeddings-stefan-it/lm-nl-opus-large-forward-v0.1.pt",
            "nl-backward": f"{aws_path}/embeddings-stefan-it/lm-nl-opus-large-backward-v0.1.pt",
            "nl-v0-forward": f"{aws_path}/embeddings-v0.4/lm-nl-large-forward-v0.1.pt",
            "nl-v0-backward": f"{aws_path}/embeddings-v0.4/lm-nl-large-backward-v0.1.pt",
            # Norwegian
            "no-forward": f"{aws_path}/embeddings-stefan-it/lm-no-opus-large-forward-v0.1.pt",
            "no-backward": f"{aws_path}/embeddings-stefan-it/lm-no-opus-large-backward-v0.1.pt",
            # Polish
            "pl-forward": f"{aws_path}/embeddings/lm-polish-forward-v0.2.pt",
            "pl-backward": f"{aws_path}/embeddings/lm-polish-backward-v0.2.pt",
            "pl-opus-forward": f"{aws_path}/embeddings-stefan-it/lm-pl-opus-large-forward-v0.1.pt",
            "pl-opus-backward": f"{aws_path}/embeddings-stefan-it/lm-pl-opus-large-backward-v0.1.pt",
            # Portuguese
            "pt-forward": f"{aws_path}/embeddings-v0.4/lm-pt-forward.pt",
            "pt-backward": f"{aws_path}/embeddings-v0.4/lm-pt-backward.pt",
            # Pubmed
            "pubmed-forward": f"{aws_path}/embeddings-v0.4.1/pubmed-2015-fw-lm.pt",
            "pubmed-backward": f"{aws_path}/embeddings-v0.4.1/pubmed-2015-bw-lm.pt",
            # Slovenian
            "sl-forward": f"{aws_path}/embeddings-stefan-it/lm-sl-opus-large-forward-v0.1.pt",
            "sl-backward": f"{aws_path}/embeddings-stefan-it/lm-sl-opus-large-backward-v0.1.pt",
            "sl-v0-forward": f"{aws_path}/embeddings-v0.3/lm-sl-large-forward-v0.1.pt",
            "sl-v0-backward": f"{aws_path}/embeddings-v0.3/lm-sl-large-backward-v0.1.pt",
            # Swedish
            "sv-forward": f"{aws_path}/embeddings-stefan-it/lm-sv-opus-large-forward-v0.1.pt",
            "sv-backward": f"{aws_path}/embeddings-stefan-it/lm-sv-opus-large-backward-v0.1.pt",
            "sv-v0-forward": f"{aws_path}/embeddings-v0.4/lm-sv-large-forward-v0.1.pt",
            "sv-v0-backward": f"{aws_path}/embeddings-v0.4/lm-sv-large-backward-v0.1.pt",
            # Tamil
            "ta-forward": f"{aws_path}/embeddings-stefan-it/lm-ta-opus-large-forward-v0.1.pt",
            "ta-backward": f"{aws_path}/embeddings-stefan-it/lm-ta-opus-large-backward-v0.1.pt",
            # CLEF HIPE Shared task
            "de-impresso-hipe-v1-forward": f"{clef_hipe_path}/de-hipe-flair-v1-forward/best-lm.pt",
            "de-impresso-hipe-v1-backward": f"{clef_hipe_path}/de-hipe-flair-v1-backward/best-lm.pt",
            "en-impresso-hipe-v1-forward": f"{clef_hipe_path}/en-flair-v1-forward/best-lm.pt",
            "en-impresso-hipe-v1-backward": f"{clef_hipe_path}/en-flair-v1-backward/best-lm.pt",
            "fr-impresso-hipe-v1-forward": f"{clef_hipe_path}/fr-hipe-flair-v1-forward/best-lm.pt",
            "fr-impresso-hipe-v1-backward": f"{clef_hipe_path}/fr-hipe-flair-v1-backward/best-lm.pt",
        }

        if type(model) == str:

            # load model if in pretrained model map
            if model.lower() in self.PRETRAINED_MODEL_ARCHIVE_MAP:
                base_path = self.PRETRAINED_MODEL_ARCHIVE_MAP[model.lower()]

                # Fix for CLEF HIPE models (avoid overwriting best-lm.pt in cache_dir)
                if "impresso-hipe" in model.lower():
                    cache_dir = cache_dir / model.lower()
                model = cached_path(base_path, cache_dir=cache_dir)

            elif replace_with_language_code(model) in self.PRETRAINED_MODEL_ARCHIVE_MAP:
                base_path = self.PRETRAINED_MODEL_ARCHIVE_MAP[
                    replace_with_language_code(model)
                ]
                model = cached_path(base_path, cache_dir=cache_dir)

            elif not Path(model).exists():
                raise ValueError(
                    f'The given model "{model}" is not available or is not a valid path.'
                )

        from flair.models import LanguageModel

        if type(model) == LanguageModel:
            self.lm: LanguageModel = model
            self.name = f"Task-LSTM-{self.lm.hidden_size}-{self.lm.nlayers}-{self.lm.is_forward_lm}"
        else:
            self.lm: LanguageModel = LanguageModel.load_language_model(model)
            self.name = str(model)

        # embeddings are static if we don't do finetuning
        self.fine_tune = fine_tune
        self.static_embeddings = not fine_tune

        self.is_forward_lm: bool = self.lm.is_forward_lm
        self.with_whitespace: bool = with_whitespace
        self.tokenized_lm: bool = tokenized_lm
        self.chars_per_chunk: int = chars_per_chunk

        # embed a dummy sentence to determine embedding_length
        dummy_sentence: Sentence = Sentence()
        dummy_sentence.add_token(Token("hello"))
        embedded_dummy = self.embed(dummy_sentence)
        self.__embedding_length: int = len(
            embedded_dummy[0].get_token(1).get_embedding()
        )

        # set to eval mode
        self.eval()

    def train(self, mode=True):

        # make compatible with serialized models (TODO: remove)
        if "fine_tune" not in self.__dict__:
            self.fine_tune = False
        if "chars_per_chunk" not in self.__dict__:
            self.chars_per_chunk = 512

        if not self.fine_tune:
            pass
        else:
            super(FlairEmbeddings, self).train(mode)

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        # make compatible with serialized models (TODO: remove)
        if "with_whitespace" not in self.__dict__:
            self.with_whitespace = True
        if "tokenized_lm" not in self.__dict__:
            self.tokenized_lm = True

        # gradients are enable if fine-tuning is enabled
        gradient_context = torch.enable_grad() if self.fine_tune else torch.no_grad()

        with gradient_context:

            # if this is not possible, use LM to generate embedding. First, get text sentences
            text_sentences = [sentence.to_tokenized_string() for sentence in sentences] if self.tokenized_lm \
                else [sentence.to_plain_string() for sentence in sentences]

            start_marker = self.lm.document_delimiter if "document_delimiter" in self.lm.__dict__ else '\n'
            end_marker = " "

            # get hidden states from language model
            all_hidden_states_in_lm = self.lm.get_representation(
                text_sentences, start_marker, end_marker, self.chars_per_chunk
            )

            if not self.fine_tune:
                all_hidden_states_in_lm = all_hidden_states_in_lm.detach()

            # take first or last hidden states from language model as word representation
            for i, sentence in enumerate(sentences):
                sentence_text = sentence.to_tokenized_string() if self.tokenized_lm else sentence.to_plain_string()

                offset_forward: int = len(start_marker)
                offset_backward: int = len(sentence_text) + len(start_marker)

                for token in sentence.tokens:

                    offset_forward += len(token.text)
                    if self.is_forward_lm:
                        offset_with_whitespace = offset_forward
                        offset_without_whitespace = offset_forward - 1
                    else:
                        offset_with_whitespace = offset_backward
                        offset_without_whitespace = offset_backward - 1

                    # offset mode that extracts at whitespace after last character
                    if self.with_whitespace:
                        embedding = all_hidden_states_in_lm[offset_with_whitespace, i, :]
                    # offset mode that extracts at last character
                    else:
                        embedding = all_hidden_states_in_lm[offset_without_whitespace, i, :]

                    if self.tokenized_lm or token.whitespace_after:
                        offset_forward += 1
                        offset_backward -= 1

                    offset_backward -= len(token.text)

                    # only clone if optimization mode is 'gpu'
                    if flair.embedding_storage_mode == "gpu":
                        embedding = embedding.clone()

                    token.set_embedding(self.name, embedding)

            del all_hidden_states_in_lm

        return sentences

    def __str__(self):
        return self.name


class PooledFlairEmbeddings(TokenEmbeddings):
    def __init__(
        self,
        contextual_embeddings: Union[str, FlairEmbeddings],
        pooling: str = "min",
        only_capitalized: bool = False,
        **kwargs,
    ):

        super().__init__()

        # use the character language model embeddings as basis
        if type(contextual_embeddings) is str:
            self.context_embeddings: FlairEmbeddings = FlairEmbeddings(
                contextual_embeddings, **kwargs
            )
        else:
            self.context_embeddings: FlairEmbeddings = contextual_embeddings

        # length is twice the original character LM embedding length
        self.embedding_length = self.context_embeddings.embedding_length * 2
        self.name = self.context_embeddings.name + "-context"

        # these fields are for the embedding memory
        self.word_embeddings = {}
        self.word_count = {}

        # whether to add only capitalized words to memory (faster runtime and lower memory consumption)
        self.only_capitalized = only_capitalized

        # we re-compute embeddings dynamically at each epoch
        self.static_embeddings = False

        # set the memory method
        self.pooling = pooling
        if pooling == "mean":
            self.aggregate_op = torch.add
        elif pooling == "fade":
            self.aggregate_op = torch.add
        elif pooling == "max":
            self.aggregate_op = torch.max
        elif pooling == "min":
            self.aggregate_op = torch.min

    def train(self, mode=True):
        super().train(mode=mode)
        if mode:
            # memory is wiped each time we do a training run
            print("train mode resetting embeddings")
            self.word_embeddings = {}
            self.word_count = {}

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        self.context_embeddings.embed(sentences)

        # if we keep a pooling, it needs to be updated continuously
        for sentence in sentences:
            for token in sentence.tokens:

                # update embedding
                local_embedding = token._embeddings[self.context_embeddings.name].cpu()

                # check token.text is empty or not
                if token.text:
                    if token.text[0].isupper() or not self.only_capitalized:

                        if token.text not in self.word_embeddings:
                            self.word_embeddings[token.text] = local_embedding
                            self.word_count[token.text] = 1
                        else:
                            aggregated_embedding = self.aggregate_op(
                                self.word_embeddings[token.text], local_embedding
                            )
                            if self.pooling == "fade":
                                aggregated_embedding /= 2
                            self.word_embeddings[token.text] = aggregated_embedding
                            self.word_count[token.text] += 1

        # add embeddings after updating
        for sentence in sentences:
            for token in sentence.tokens:
                if token.text in self.word_embeddings:
                    base = (
                        self.word_embeddings[token.text] / self.word_count[token.text]
                        if self.pooling == "mean"
                        else self.word_embeddings[token.text]
                    )
                else:
                    base = token._embeddings[self.context_embeddings.name]

                token.set_embedding(self.name, base)

        return sentences

    def embedding_length(self) -> int:
        return self.embedding_length

    def __setstate__(self, d):
        self.__dict__ = d

        if flair.device != 'cpu':
            for key in self.word_embeddings:
                self.word_embeddings[key] = self.word_embeddings[key].cpu()
