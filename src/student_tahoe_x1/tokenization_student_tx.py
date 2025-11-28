from transformers import PreTrainedTokenizer
from typing import List, Optional, Dict, Union, Tuple
import json
from pathlib import Path

# Assuming tahoe_x1.tokenizer.gene_tokenizer is importable
# We might need to adjust the import path based on the final project structure
from tahoe_x1.tokenizer.gene_tokenizer import GeneVocab

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json"}

class StudentTXTokenizer(PreTrainedTokenizer):
    """
    Constructs a StudentTX tokenizer.
    This tokenizer wraps the custom GeneVocab class to be compatible with HuggingFace's PreTrainedTokenizer.

    Args:
        vocab_file (`str`): Path to the vocabulary file.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is replaced by this
            token.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (`str`, *optional*, defaults to `"<cls>"`):
            The classifier token.
        mask_token (`str`, *optional*, defaults to `"<mask>"`):
            The mask token.
        **kwargs: Additional keyword arguments.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: str,
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        cls_token: str = "<cls>",
        mask_token: str = "<mask>",
        **kwargs,
    ):
        self.vocab_file = vocab_file
        self.vocab = GeneVocab.from_file(vocab_file)

        # Ensure the unk_token is in the vocabulary before setting it as default
        if unk_token not in self.vocab:
            self.vocab.append_token(unk_token)

        # Set special tokens based on GeneVocab's understanding
        self.vocab.set_default_token(unk_token)
        self.vocab.pad_token = pad_token
        
        super().__init__(
            unk_token=unk_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> Dict[str, int]:
        return self.vocab.get_stoi()

    def _tokenize(self, text: str) -> List[str]:
        """Converts a token string to a sequence of tokens (if text is a single gene name)"""
        # GeneVocab expects a list of gene names to convert to IDs, not a single string token.
        # This method is usually for breaking down a sentence into words.
        # For single-cell data, 'text' is typically a gene name.
        # So, we just return the text as a list of one token.
        return [text]

    def _convert_token_to_id(self, token: str) -> int:
        """Converts a token (str) in an id (integer) using the vocab."""
        return self.vocab[token]

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an id (integer) in a token (str) using the vocab."""
        return self.vocab.index_to_token[index]

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Converts a sequence of tokens (string) in a single string."""
        return " ".join(tokens)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Saves the vocabulary (the token-to-id mapping) of the tokenizer to a file.
        """
        if not Path(save_directory).is_dir():
            Path(save_directory).mkdir(parents=True, exist_ok=True)

        vocab_file = Path(save_directory) / (
            (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        self.vocab.save_json(vocab_file)
        return (str(vocab_file),)

    def _prepare_for_model(self, *args, **kwargs):
        # This method is called internally by __call__ (and encode_plus, batch_encode_plus)
        # to add special tokens and create attention masks.
        # We'll rely on the default implementation for now, or customize if needed.
        # For gene data, we might not need complex attention masks beyond simple padding.
        return super()._prepare_for_model(*args, **kwargs)

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0
        # For now, we only handle single sequence. Extend if needed for pair sequences.
        raise NotImplementedError("StudentTXTokenizer does not currently support pair sequences.")

    def get_special_tokens_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False) -> List[int]:
        """
        Retrieves sequence of strings representing the token_ids_0 tokens and adds all special tokens also.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )
        
        if token_ids_1 is not None:
            raise NotImplementedError("StudentTXTokenizer does not currently support pair sequences for special token mask.")
        
        # Assume <cls> is the only special token added at the beginning
        return [1] + ([0] * len(token_ids_0))

    def create_token_type_ids_from_sequences(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task.
        """
        if token_ids_1 is not None:
            raise NotImplementedError("StudentTXTokenizer does not currently support pair sequences for token type ids.")
        
        return [0] * (len(token_ids_0) + 1) # +1 for CLS token
