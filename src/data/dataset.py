import torch
from torch.utils.data import Dataset

def causal_mask(size):
    """
    Create a causal mask for the transformer model.
    """
    mask = torch.triu(torch.ones(size, size), diagonal=1).type(torch.int)
    return mask == 0

class LanguageModelingDataset(Dataset):
    def __init__(self, ds, tokenizer, seq_len, preprocessed=False):
        """
        Initialize the dataset.

        :param ds: Dataset source (list of dicts or preprocessed data).
        :param tokenizer: Tokenizer object for encoding text.
        :param seq_len: Maximum sequence length for padding/truncation.
        :param preprocessed: If True, assumes ds is preprocessed data.
        """
        self.ds = ds
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.preprocessed = preprocessed

        # Token IDs for special tokens
        self.sos_token = torch.tensor([tokenizer.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer.token_to_id('[PAD]')], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        """
        Get a single item from the dataset.
        :param index: Index of the data item.
        :return: Dictionary containing input tokens, masks, and labels.
        """
        if self.preprocessed:
            # Load preprocessed data directly
            item = self.ds[index]
            input_tokens = torch.tensor(item["input_tokens"], dtype=torch.long)
            label = torch.tensor(item["label"], dtype=torch.long)
        else:
            # Inline tokenization and processing
            text = self.ds[index]['text']
            tokens = self.tokenizer.encode(text).ids

            num_padding_tokens = self.seq_len - len(tokens) - 2
            if num_padding_tokens < 0:
                raise ValueError('Sequence too long')

            input_tokens = torch.cat(
                [
                    self.sos_token,
                    torch.tensor(tokens, dtype=torch.int64),
                    self.eos_token,
                    torch.tensor([self.pad_token] * num_padding_tokens, dtype=torch.int64)
                ]
            )

            label = torch.cat(
                [
                    torch.tensor(tokens, dtype=torch.int64),
                    self.eos_token,
                    torch.tensor([self.pad_token] * num_padding_tokens, dtype=torch.int64)
                ]
            )

        return {
            "input_tokens": input_tokens,
            "attention_mask": (input_tokens != self.pad_token).int(),  # Binary mask for valid tokens
            "causal_mask": causal_mask(input_tokens.size(0)),  # Causal mask for self-attention
            "label": label
        }