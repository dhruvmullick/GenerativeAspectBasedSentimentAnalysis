from transformers import LogitsProcessor
import torch
import numpy as np


class CopyWordLogitsProcessor(LogitsProcessor):

    ### We assume only one sentence is passed for generation in the given batch.
    def __init__(self, original_input_ids, attention_mask, tokenizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.original_input_ids = original_input_ids[0].cpu()
        self.attention_mask = attention_mask[0].cpu()
        self.whitelisted_token_ids = self.original_input_ids[np.where(self.attention_mask == 1)]
        self.whitelisted_token_ids = self.whitelisted_token_ids.cuda()
        self.tokenizer = tokenizer

        # print("Concatenating...")

        self.whitelisted_token_ids = torch.cat((self.whitelisted_token_ids,
                                                torch.tensor(tokenizer.encode('positive')).cuda()), 0)
        self.whitelisted_token_ids = torch.cat((self.whitelisted_token_ids,
                                                torch.tensor(tokenizer.encode('negative')).cuda()), 0)
        self.whitelisted_token_ids = torch.cat((self.whitelisted_token_ids,
                                                torch.tensor(tokenizer.encode('neutral')).cuda()), 0)
        self.whitelisted_token_ids = torch.cat((self.whitelisted_token_ids,
                                                torch.tensor(tokenizer.encode('<sep>')).cuda()), 0)
        self.whitelisted_token_ids = torch.cat((self.whitelisted_token_ids,
                                                torch.tensor(tokenizer.encode('<lang>')).cuda()), 0)
        self.whitelisted_token_ids = torch.unique(self.whitelisted_token_ids)

        # print("Concatenated...")

        self.mask = torch.ones(len(tokenizer.get_vocab().values()))
        self.mask[self.whitelisted_token_ids] = 0
        self.mask = self.mask.bool().cuda()

        # print("Generated mask...")

    def __call__(self, input_ids, scores):

        # print("Called...")

        scores = scores.cuda()
        scores = scores.masked_fill(self.mask, -float("inf"))

        # print("Masked...")

        return scores
