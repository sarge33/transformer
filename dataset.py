from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        """
        A PyTorch Dataset class for bilingual data used in machine translation. It tokenizes source-target language pairs, 
        adds special tokens (such as SOS, EOS, and PAD), and prepares input-output tensors for training a Transformer model.

        Args:
            ds (list): A list of dictionaries containing 'translation' pairs of source and target sentences.
            tokenizer_src (Any): The tokenizer for the source language.
            tokenizer_tgt (Any): The tokenizer for the target language.
            src_lang (str): The code or identifier for the source language.
            tgt_lang (str): The code or identifier for the target language.
            seq_len (int): The maximum sequence length for both source and target sentences.

        Attributes:
            sos_token (torch.Tensor): The tensor representing the start-of-sequence token.
            eos_token (torch.Tensor): The tensor representing the end-of-sequence token.
            pad_token (torch.Tensor): The tensor representing the padding token.

        Methods:
            __len__(): Returns the number of samples in the dataset.
            __getitem__(index): Returns a single sample from the dataset, including encoder input, decoder input, 
                                masks, labels, and the original texts.
        """
    
        super().__init__()
        self.ds = ds  # Dataset containing translation pairs
        self.tokenizer_src = tokenizer_src  # Source language tokenizer
        self.tokenizer_tgt = tokenizer_tgt  # Target language tokenizer
        self.src_lang = src_lang  # Source language identifier
        self.tgt_lang = tgt_lang  # Target language identifier
        self.seq_len = seq_len  # Maximum sequence length for padding/truncation

        # Special tokens
        # Start of sentence (SOS)
        self.sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64)
        # End of sentence (EOS)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64)
        # Padding token (PAD)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype=torch.int64)

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.ds)
    
    def __getitem__(self, index: Any) -> Any:
        """
        Retrieves a single sample from the dataset and processes it into the required format.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the following keys:
                - "encoder_input" (torch.Tensor): The input sequence for the encoder with [SOS], [EOS], and [PAD].
                - "decoder_input" (torch.Tensor): The input sequence for the decoder with [SOS] and [PAD].
                - "encoder_mask" (torch.Tensor): Mask for the encoder to ignore [PAD] tokens.
                - "decoder_mask" (torch.Tensor): Mask for the decoder, combining padding and causal masks.
                - "label" (torch.Tensor): The target sequence for training with [EOS] and [PAD].
                - "src_text" (str): The original source text.
                - "tgt_text" (str): The original target text.
        """
        
        # Retrieve the source and target texts
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # Tokenize the source and target texts
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids  # Tokenized source text
        dec_input_tokens = self.tokenizer_src.encode(tgt_text).ids  # Tokenized target text

        # Calculate the number of padding tokens for source and target sequences
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # 2 is for [SOS] and [EOS]
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1  # 1 is for [SOS]

        # Check if the sentence length exceeds the sequence length limit
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('Sentence is too long')

        # Construct the encoder input with [SOS], tokenized source, [EOS], and padding
        encoder_input = torch.cat(
            [
                self.sos_token,  # Start of sentence
                torch.tensor(enc_input_tokens, dtype=torch.int64),  # Tokenized source text
                self.eos_token,  # End of sentence
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)  # Padding tokens
            ]
        )

        # Construct the decoder input with [SOS], tokenized target, and padding
        decoder_input = torch.cat(
            [
                self.sos_token,  # Start of sentence
                torch.tensor(dec_input_tokens, dtype=torch.int64),  # Tokenized target text
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)  # Padding tokens
            ]
        )

        # Construct the target label by adding [EOS] to the decoder input (for teacher forcing)
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),  # Tokenized target text
                self.eos_token,  # End of sentence
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)  # Padding tokens
            ]
        )

        # Ensure all tensors have the correct sequence length
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        # Create the encoder and decoder masks to handle padding and self-attention
        return {
            "encoder_input": encoder_input,  # Input to the encoder (seq_len)
            "decoder_input": decoder_input,  # Input to the decoder (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),  # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & casual_mask(decoder_input.size(0)),  # (1, seq_len) & (1, seq_len, seq_len)
            "label": label,  # The target output (seq_len)
            "src_text": src_text,  # Original source text
            "tgt_text": tgt_text   # Original target text
        }

def casual_mask(size: int) -> torch.Tensor:
    """
    Creates a causal mask for the decoder to prevent attending to future positions.

    Args:
        size (int): The length of the sequence.

    Returns:
        torch.Tensor: A mask of shape (1, size, size) with ones below the diagonal and zeros above.
    """
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)  # Upper triangular mask
    return mask == 0  # Return a boolean mask with 1s where allowed and 0s where masked


# class BilingualDataset(Dataset):

#     def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
#         super().__init__()
#         self.ds = ds
#         self.tokenizer_src = tokenizer_src
#         self.tokenizer_tgt = tokenizer_tgt
#         self.src_lang = src_lang
#         self.tgt_lang = tgt_lang
#         self.seq_len = seq_len

#         # start of sentence token
#         self.sos_token = torch.Tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64)
#         # end of sentence token
#         self.eos_token = torch.Tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64)
#         # padding of sentence token
#         self.pad_token = torch.Tensor([tokenizer_src.token_to_id('[PAD]')], dtype=torch.int64)

#     def __len__(self):
#         return len(self.ds)
    
#     def __getitem__(self, index: Any) -> Any:
#         src_target_pair = self.ds[index]
#         src_text = src_target_pair['translation'][self.src_lang]
#         tgt_text = src_target_pair['translation'][self.tgt_lang]

#         enc_input_tokens = self.tokenizer_src.encode(src_text).ids
#         dec_input_tokens = self.tokenizer_src.encode(tgt_text).ids

#         enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # 2 is from [SOS] and [EOS]
#         dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1  # 1 is only for [SOS]
        
#         if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
#             raise ValueError('Sentence is too long')
        
#         # Add SOS and EOS to the source text
#         encoder_input = torch.cat(
#             [
#                 self.sos_token,
#                 torch.tensor(enc_input_tokens, dtype=torch.int64),
#                 self.eos_token,
#                 torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
#             ]
#         )

#         # Add SOS to the decoder text
#         decoder_input = torch.cat(
#             [
#                 self.sos_token,
#                 torch.tensor(dec_input_tokens, dtype=torch.int64),
#                 torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
#             ]
#         )

#         # Add EOS to the label (what we expect as output from the decoder)
#         label = torch.cat(
#             [
#                 torch.tensor(dec_input_tokens, dtype=torch.int64),
#                 self.eos_token,
#                 torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
#             ]
#         )

#         assert encoder_input.size(0) == self.seq_len
#         assert decoder_input.size(0) == self.seq_len
#         assert label.size(0) == self.seq_len

#         return {
#             "encoder_input": encoder_input,   # (seq_len)
#             "decoder_input": decoder_input,   # (seq_len)
#             "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),   # (1, 1, seq_len)
#             "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & casual_mask(decoder_input.size(0)),  # (1, seq_len) & (1, seq_len, seq_len)
#             "label": label, # (seq_len)
#             "src_text": src_text,
#             "tgt_text": tgt_text
#         }
    
# def casual_mask(size):
#     mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
#     return mask == 0