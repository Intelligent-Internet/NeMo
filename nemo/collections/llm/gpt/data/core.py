# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from typing import TYPE_CHECKING, Optional
from nemo.lightning.base import NEMO_DATASETS_CACHE
from nemo.collections.common.tokenizers import TokenizerSpec
from nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_dataset import GPTSFTDataset
from nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_chat_dataset import GPTSFTChatDataset
from datasets import load_dataset
import torch

def get_dataset_root(name: str) -> Path:
    output = Path(NEMO_DATASETS_CACHE) / name
    output.mkdir(parents=True, exist_ok=True)
    return output



class ChatMLDataset(GPTSFTChatDataset):
    
    def _load_dataset(self):
        self.indexed_dataset = load_dataset(
            self.file_path, 
            keep_in_memory=True
        )['train']
    
    def __len__(self):
        return len(self.indexed_dataset)
    
    def __getitem__(self, idx):
        sample = self.indexed_dataset[idx]
        return self._process_example(sample)
    
    def _process_example(self, example):
        """
        Create an example by concatenating text and answer.
        Truncation is carried out when needed, but it is performed only on the prompt side.
        BOS, EOS, and SEP, are added if specified.
        """
        batched_token_ids = []
        batched_masks = []
        conversation = example["messages"]
        if conversation[0]["role"] == "system":
            system_text = conversation[0]["content"]
            conversation = conversation[1:]
        else:
            system_text = "You are a helpful assistant!"
        system_text = f"<im_start>system\n{system_text}<im_end>"
        system_token_ids = self.tokenizer.text_to_ids(system_text)
        system_masks = [0] * len(system_token_ids)
        batched_token_ids.extend(system_token_ids)
        batched_masks.extend(system_masks)

        for (i, item) in enumerate(conversation):
            if item["role"] == "user":
                full_text = f"<im_start>user\n{item['content']}<im_end>"
                token_ids = self.tokenizer.text_to_ids(full_text)
                masks = [0] * len(token_ids)
                batched_token_ids.extend(token_ids)
                batched_masks.extend(masks)
                continue

            if item["role"] == "assistant":
                full_text = f"<im_start>assistant\n{item['content']}<im_end>"
                token_ids = self.tokenizer.text_to_ids(full_text)
                masks = [1] * len(token_ids)
                batched_token_ids.extend(token_ids)
                batched_masks.extend(masks)
                continue

        assert len(batched_token_ids) == len(batched_masks)
        processed_batch = {
            "input_ids": [batched_token_ids],
            "mask": [batched_masks],
        }
        return processed_batch

    def collate_fn(self, batch):
        # return batch
        input_ids = [item[:-1] for one_batch in batch for item in one_batch["input_ids"]]
        labels = [item[1:] for one_batch in batch for item in one_batch["input_ids"]]
        loss_mask = [item[1:] for one_batch in batch for item in one_batch["mask"]]
        num_responses = [len(one_batch["input_ids"]) for one_batch in batch for item in one_batch["input_ids"]]
        # assert num_responses all have the same number and only one number
        assert len(set(num_responses)) == 1
        max_length = max([len(x) for x in input_ids])

        if max_length > self.max_seq_length:
            # truncate the sequences if it is longer than max_seq_length
            input_ids = [x[: self.max_seq_length] for x in input_ids]
            labels = [x[: self.max_seq_length] for x in labels]
            loss_mask = [x[: self.max_seq_length] for x in loss_mask]

        # increase max length to nearest multiple of 4 or 8
        if self.pad_to_max_length:
            max_length = self.max_seq_length
        else:
            max_length = min(self.max_seq_length, self._ceil_to_nearest(max_length, 8))
        assert max_length <= self.max_seq_length

        attention_mask = [self._create_attention_mask(max_length) for _ in batch]
        attention_mask = torch.stack(attention_mask)
        position_ids = [list(range(max_length)) for _ in batch]
        position_ids = torch.LongTensor(position_ids)
        input_ids = torch.LongTensor(
            self._collate_item(input_ids, max_length=max_length, pad_id=self.tokenizer.eos_id)
        )
        labels = torch.LongTensor(self._collate_item(labels, max_length=max_length, pad_id=self.tokenizer.eos_id))
        loss_mask = torch.LongTensor(self._collate_item(loss_mask, max_length=max_length, pad_id=0))
        num_responses = torch.LongTensor(num_responses)

        processed_batch = {
            "tokens": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
            "num_responses": num_responses,
        }

        return processed_batch

def create_sft_dataset(
    path: Path,
    tokenizer: "TokenizerSpec",
    seq_length: int = 2048,
    add_bos: bool = False,
    add_eos: bool = True,
    add_sep: bool = False,
    seed: int = 1234,
    label_key: str = 'output',
    answer_only_loss: bool = True,
    truncation_field: str = 'input',
    pad_to_max_length: bool = False,
    index_mapping_dir: Optional[str] = None,
    prompt_template: str = '{input} {output}',
    truncation_method: str = 'right',
    memmap_workers: int = 2,
    hf_dataset: bool = False,
    global_sample_mapping: bool = False,
    pack_metadata_file_path: Path = None,
    pad_cu_seqlens: bool = False,
    chat: bool = False,
    **kwargs,
) -> "GPTSFTDataset":
    """
    Create the dataset class (GPTSFTDataset, GPTSFTChatDataset or GPTSFTPackedDataset)
    """

    gpt_sft_dataset_kwargs = {
        'file_path': str(path),
        'tokenizer': tokenizer,
        'max_seq_length': seq_length,
        'memmap_workers': memmap_workers,
        'hf_dataset': hf_dataset,
        'global_sample_mapping': global_sample_mapping,
        'add_bos': add_bos,
        'add_eos': add_eos,
        'add_sep': add_sep,
        'seed': seed,
        'label_key': label_key,
        'answer_only_loss': answer_only_loss,
        'truncation_field': truncation_field,
        'pad_to_max_length': pad_to_max_length,
        'index_mapping_dir': index_mapping_dir,
        'prompt_template': prompt_template,
        'truncation_method': truncation_method,
    }
    
    if kwargs.get('chat', True):
        return ChatMLDataset(**gpt_sft_dataset_kwargs, **kwargs)

    if chat:
        from nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_chat_dataset import GPTSFTChatDataset

        return GPTSFTChatDataset(
            **gpt_sft_dataset_kwargs,
            **kwargs,
        )
    elif path.suffix == '.npy':
        from nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_dataset import GPTSFTPackedDataset

        return GPTSFTPackedDataset(
            pack_metadata_file_path=pack_metadata_file_path,
            pad_cu_seqlens=pad_cu_seqlens,
            **gpt_sft_dataset_kwargs,
            **kwargs,
        )
    else:
        from nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_dataset import GPTSFTDataset

        return GPTSFTDataset(
            **gpt_sft_dataset_kwargs,
            **kwargs,
        )
