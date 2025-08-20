from patch import replace_llama_attn_with_xformers_attn
replace_llama_attn_with_xformers_attn()
print("using xformers")

import copy
import glob
import os
import time
import re
from collections import deque
from pathlib import Path
import minigrid
import gymnasium as gym
import gym_cards
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict
torch.backends.cuda.enable_mem_efficient_sdp(False)  #disabling cutlass not working on small gpus
torch.backends.cuda.enable_flash_sdp(False)

from a2c_ppo_acktr import algo, utils, rl_utils

from a2c_ppo_acktr.rl_utils import get_prompt, text_projection, text_projection_pr
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.llava_interface import qwen_process, qwen_batch_process, format_data_sft, qwen_process_multiImg
from a2c_ppo_acktr.model import VLMPolicy, VLMValue, QwenVLMValue, QwenVLMPolicy, QwenTempPredictor
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr.llava_interface import llava_evaluate, llava_generate, qwen_batch_process_multiIm
from a2c_ppo_acktr.llava_interface import init_pretrained_model, find_all_linear_names, load_lora_model

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.model import LlavaLlamaForCausalLM
from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM

from typing import Dict, Optional, Sequence, List
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLProcessor

from SFT.trainer import VLMTrainer

from torch.utils.data import Dataset, DataLoader
import math
import random
from functools import partial
from typing import List, Optional
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoImageProcessor, AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration, HfArgumentParser, Qwen2_5_VLForConditionalGeneration
import transformers

import json
from tqdm import tqdm

import accelerate
from accelerate.state import AcceleratorState

from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
from torch.utils.data import Dataset, random_split
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings("ignore")

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def cosine_schedule(step, T_max, eta_min, base_lr):
    """
    Returns the LR multiplier at step `step`, where:
    - T_max is the total number of steps (like in CosineAnnealingLR)
    - eta_min is the final LR (absolute)
    - base_lr is the starting LR (absolute)

    Output is a multiplier relative to base_lr.
    """
    if step >= T_max:
        return eta_min / base_lr  # stays flat after schedule ends
    cosine_decay = 0.5 * (1 + math.cos(math.pi * step / T_max))
    return (eta_min + (base_lr - eta_min) * cosine_decay) / base_lr

def get_trainable_params( model, return_with_names=False):
        if return_with_names:
            return filter(lambda p: p[1].requires_grad, model.named_parameters())
        else:
            return filter(lambda p: p.requires_grad, model.parameters())


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    cuda_deterministic: bool = False
    tf32 = False
    seed: int = 1

    lora_enable: bool = False
    lora_r: int = 128 #64
    lora_alpha: int = 256 #16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)





class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 processor,
                 data_args: DataArguments,
                 train=False):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.processor = processor
        self.list_data_dict = list_data_dict
        self.train = train
        self.data_args = data_args
        self.query = (
                "You're an expert in 2D grid-based games guiding a player by scoring his each step based. "
                "The player is shown by cyan triangle.The tip (pointy end) of the triangle is the direction the player is facing, the flat side is the back. In the game the player must pick up a key to unlock the door. The square with a minus (-) and blue or yellow color is the closed door. The player shuold reach the pink goal tile to win. "
                "At each step the player takes one of these actions ['Turn left': turns direction to left, 'Turn right': turns direction to right, 'Move forward': take one step to front, 'Pick up': picks key only if key was in front of it in first image, 'Toggle': toggle door only if door was infront of it in first image]. "
                "You are given two images of the game environment. "
                "The first image shows the state *before* taking action: {action}. "
                "The second image shows the state *after* that action. "
                "Analyze what has changed in the second image compared to the first. Did the action have any useful effect? "
                "Rate the usefulness of the action from 0 (useless) to 10 (very useful), based only on the visual evidence in the images.\n "
                "**Scoring rubric (0 – 10):** \n"
                "- **10:** Successful interaction or completion (valid Pick up, valid Toggle that opens needed door, or reaching goal with prerequisites met). \n"
                "- **8 – 9:** Clear improvement: moved closer to the current objective by 1+ tile **or** turned to face it directly (from misaligned to aligned). \n"
                "- **5 – 7:** Minor but real progress: slight angle improvement or small distance improvement that sets up a good next step. \n"
                "- **1 – 2:** Neutral/ineffective: no distance/angle improvement; sideways movement; turn that keeps facing irrelevant space. \n"
                "- **0:** Counterproductive: increased distance, turned away from the objective, moved toward a door while a key exists elsewhere, or attempted invalid Pick up/Toggle. \n"
                "Respond strictly with a JSON object:\n"
                "{{\n"
                "\"thoughts\": \"Describe ONLY the changes between before and after images.\",\n"
                "\"score\": your_score\n"
                "}}"
            ) #TODO the main query where the {action} will be put in
        
        self.filter_data()
        
        self.transform = T.ToTensor() 

    def __len__(self):
        return len(self.list_data_dict)
    
    def filter_data(self):
        cleaned_data = []
        for elem in self.list_data_dict:
            try:
                # If elem is a string, try parsing
                string = elem['response']
                match = re.search(r"```(?:json)?\n(.*?)\n```", string, re.DOTALL)
                if match:
                    cleaned_data.append(elem)
            except:
                # skip this element
                
                continue

        self.list_data_dict = cleaned_data

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        # if isinstance(i, int):
        #     sources = [sources]
        # assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        image_folder = self.data_args.image_folder
        
        img1_file = sources['before_img']
        img2_file = sources['after_img']
        img1 = Image.open(os.path.join(image_folder, img1_file)) #.convert('RGB')
        img2 = Image.open(os.path.join(image_folder, img2_file))
        img1 = self.transform(img1).permute(1,2, 0).unsqueeze(0)
        img2 = self.transform(img2).permute(1,2, 0).unsqueeze(0)
        response = sources['response']
        action = sources['action']
        util_score = sources['utility_score']
        # query = (
        #         "You're an expert in 2D grid-based games guiding a player by scoring his each step based. "
        #         "The player is shown by cyan triangle.The tip (pointy end) of the triangle is the direction the player is facing, the flat side is the back. In the game the player must pick up a key to unlock the door. The square with a minus (-) and blue or yellow color is the closed door. The player shuold reach the pink goal tile to win. "
        #         "At each step the player takes one of these actions ['Turn left': turns direction to left, 'Turn right': turns direction to right, 'Move forward': take one step to front, 'Pick up': picks key only if key was in front of it in first image, 'Toggle': toggle door only if door was infront of it in first image]. "
        #         "You are given two images of the game environment. "
        #         f"The first image shows the state *before* taking action: {action}. "
        #         "The second image shows the state *after* that action. "
        #         "Analyze what has changed in the second image compared to the first. Did the action have any useful effect? "
        #         "Rate the usefulness of the action from 0 (useless) to 10 (very useful), based only on the visual evidence in the images.\n "
        #         "**Scoring rubric (0 – 10):** \n"
        #         "- **10:** Successful interaction or completion (valid Pick up, valid Toggle that opens needed door, or reaching goal with prerequisites met). \n"
        #         "- **8 – 9:** Clear improvement: moved closer to the current objective by 1+ tile **or** turned to face it directly (from misaligned to aligned). \n"
        #         "- **5 – 7:** Minor but real progress: slight angle improvement or small distance improvement that sets up a good next step. \n"
        #         "- **1 – 2:** Neutral/ineffective: no distance/angle improvement; sideways movement; turn that keeps facing irrelevant space. \n"
        #         "- **0:** Counterproductive: increased distance, turned away from the objective, moved toward a door while a key exists elsewhere, or attempted invalid Pick up/Toggle. \n"
        #         "Respond strictly with a JSON object:\n"
        #         "{\n"
        #         "\"thoughts\": \"Describe ONLY the changes between before and after images.\",\n"
        #         "\"score\": your_score\n"
        #         "}"
        #     )
        query= self.query.format(action = action)
        
        
        data_dict = {}
        data_dict["image"] = [img1, img2]
        data_dict["query"] = query
        data_dict["label"] = response
        data_dict['util_score'] = util_score
        if self.train:
            data_dict = format_data_sft(data_dict)
        #batch = self.process_mm([formatted_data_dict])
        
        return data_dict

    def process_mm(self, examples):
        """
        examples has a length of batch-size that is not fixed and can vary
        each consist of tuple: first is chat, second is random-mask
        """
        # Get the texts and images, and apply the chat template
        
        texts = [
            self.processor.apply_chat_template(example, tokenize=False) for example in examples
        ]  # Prepare texts for processing
        image_inputs = [process_vision_info(example)[0] for example in examples]  # Process the images to extract inputs

        # Tokenize the texts and process the images
        batch = self.processor(
            text=texts, images=image_inputs, return_tensors="pt", padding=True
        )  # Encode texts and images into tensors

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()  # Clone input IDs for labels
        labels[labels == self.processor.tokenizer.pad_token_id] = -100  # Mask padding tokens in labels

        # Ignore the image token index in the loss computation (model specific)
        if isinstance(self.processor, Qwen2VLProcessor):  # Check if the processor is Qwen2VLProcessor
            image_tokens = [151652, 151653, 151655]  # Specific image token IDs for Qwen2VLProcessor
        else:
            image_tokens = [self.processor.tokenizer.convert_tokens_to_ids(self.processor.image_token)]  # Convert image token to ID
        # labels, batch_slices, random_coeffs = self.assign_random_mask_weights(labels, random_masks, bad_masks)
        # pattern = torch.tensor([1018, 19366, 4], device=labels.device)   #that is fixed token id of a special token  " %generate% "
        # match = (labels.unfold(1, len(pattern), 1) == pattern).all(-1)
        # first_match = torch.where(match.any(1), match.float().argmax(1), torch.full((labels.size(0),), labels.size(1), device=labels.device))
        # first_match = torch.where(match.any(1), match.float().argmax(1), torch.full((labels.size(0),), -len(pattern), device=labels.device))
        # mask = torch.arange(labels.size(1), device=labels.device).unsqueeze(0) < (first_match + len(pattern)).unsqueeze(1)
        # Mask image token IDs in the labels
        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100  # Mask image token IDs in labels
        #labels[mask] = -100
        
        batch["labels"] = labels  # Add labels to the batch
        batch["labels"] = batch["labels"] #.to(self.device)
        batch["input_ids"] = batch['input_ids'] #.to(self.device)
        batch['attention_mask'] = batch['attention_mask'] #.to(self.device)
        batch['pixel_values'] = batch['pixel_values'] #.to(self.device)
        batch['image_grid_thw'] = batch['image_grid_thw'] #.to(self.device)
        return batch


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = [x.squeeze(0) for x in input_ids]  # from [1, seq_len] → [seq_len]
        labels    = [x.squeeze(0) for x in labels]
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=-100)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'pixel_values' in instances[0]:
            images = [instance['pixel_values'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['pixel_values'] = torch.stack(images)
            else:
                batch['pixel_values'] = images
        
        if 'image_grid_thw' in instances[0]:
            image_grids = [instance['image_grid_thw'] for instance in instances]
            
            if all(x is not None and x.shape == image_grids[0].shape for x in image_grids):
                batch['image_grid_thw'] = torch.stack(image_grids)
            else:
                batch['image_grid_thw'] = image_grids
        
        return batch

@dataclass
class Collate_fn_qwen(object):

    def __init__(self, processor):
        self.processor = processor


    def __call__(self, examples):
        """
        examples has a length of batch-size that is not fixed and can vary
        each consist of tuple: first is chat, second is random-mask
        """
        # Get the texts and images, and apply the chat template
        breakpoint()
        texts = [
            self.processor.apply_chat_template(example, tokenize=False) for example in examples
        ]  # Prepare texts for processing

        
        image_inputs = [process_vision_info(example)[0] for example in examples]  # Process the images to extract inputs

        # Tokenize the texts and process the images
        batch = self.processor(
            text=texts, images=image_inputs, return_tensors="pt", padding=True
        )  # Encode texts and images into tensors

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()  # Clone input IDs for labels
        labels[labels == self.processor.tokenizer.pad_token_id] = -100  # Mask padding tokens in labels

        # Ignore the image token index in the loss computation (model specific)
        if isinstance(self.processor, Qwen2VLProcessor):  # Check if the processor is Qwen2VLProcessor
            image_tokens = [151652, 151653, 151655]  # Specific image token IDs for Qwen2VLProcessor
        else:
            image_tokens = [self.processor.tokenizer.convert_tokens_to_ids(self.processor.image_token)]  # Convert image token to ID
        # labels, batch_slices, random_coeffs = self.assign_random_mask_weights(labels, random_masks, bad_masks)
        pattern = torch.tensor([77091], device=labels.device)   #that is fixed token id of a special token  " %generate% "
        match = (labels.unfold(1, len(pattern), 1) == pattern).all(-1)
        first_match = torch.where(match.any(1), match.float().argmax(1), torch.full((labels.size(0),), labels.size(1), device=labels.device))
        first_match = torch.where(match.any(1), match.float().argmax(1), torch.full((labels.size(0),), -len(pattern), device=labels.device))
        mask = torch.arange(labels.size(1), device=labels.device).unsqueeze(0) < (first_match + len(pattern)).unsqueeze(1)
        
        # Mask image token IDs in the labels
        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100  # Mask image token IDs in labels
        
        labels[mask] = -100
        
        batch["labels"] = labels  # Add labels to the batch
        # batch["labels"] = batch["labels"] #.to(self.device)
        # batch["input_ids"] = batch['input_ids'] #.to(self.device)
        # batch['attention_mask'] = batch['attention_mask'] #.to(self.device)
        # batch['pixel_values'] = batch['pixel_values'] #.to(self.device)
        # batch['image_grid_thw'] = batch['image_grid_thw'] #.to(self.device)
        return batch
    
def eval_collate_fn(batch):
    
    util_scores = []
    labels = []
    queries = []
    images = []
    for input in batch:
        util_scores.append(input['util_score'])
        labels.append(input['label'])
        queries.append(input['query'])
        images.append(input['image'])

    return queries, images, labels, util_scores

def make_supervised_data_module(processor,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    full_dataset = LazySupervisedDataset(processor=processor,
                                data_path=data_args.data_path,
                                data_args=data_args)
    train_size = int(0.9 * len(full_dataset))  # 90% train
    test_size = len(full_dataset) - train_size

    train_dataset, eval_dataset = random_split(
        full_dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)  # reproducibility
    )
    data_collator = Collate_fn_qwen(processor=processor) #DataCollatorForSupervisedDataset(tokenizer=processor.tokenizer)
    #TODO  add eval_dataset as well
    return dict(train_dataset=train_dataset,
                eval_dataset = eval_dataset,
                #eval_dataset=None,
                data_collator = data_collator,
                )



def qwen_calc_utility_batch(value_model, processor, text, images):
    input = qwen_batch_process_multiIm(processor, text, images)
    base = value_model.base
    input = input.to(base.device)
    input_ids = input.input_ids
    with torch.inference_mode():
        outputs = base.generate(
        **input,
        do_sample=False,
        temperature=0.2,#args.temperature,
        num_beams=1,
        max_new_tokens=1024,
        use_cache=True,
        output_scores=True,
        output_hidden_states=True,
        return_dict_in_generate=True,
        pad_token_id=processor.tokenizer.eos_token_id,)
        output_ids = outputs['sequences'] 
    
    output_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, output_ids)
                ]
    outputs = processor.batch_decode(
                        output_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )
    return outputs



def calc_utility_batch(value_model, processor, images, INPUT_IDS):
        
        outputs = qwen_calc_utility_batch(value_model = value_model,
                                                        processor = processor,
                                                        text = INPUT_IDS,
                                                        images=images,
                                                        )
        action_scores = []
        for output in outputs:
            string = output
            string = string.lower()
            try:
                match = re.search(r"```(?:json)?\n(.*?)\n```", string, re.DOTALL)
                if match:
                    string = match.group(1)
                else:
                    string = string.strip()
                response_json = json.loads(string)
                action_score = response_json["score"]
            except:
                action_score = -1
            action_scores.append(action_score)
        return action_scores, outputs[0]


def evaluate(base, evalset, processor):


    # lora_weights = torch.load("/home/bahaduri/RL4VLM/outputs/sft/model_6.0.checkpoint", map_location='cpu')
    
    # #lora_weights = {k.replace("value_model.", "", 1): v for k, v in lora_weights.items() if k.startswith("value_model.")}
    
    # missing_keys, unexpected_keys = base.load_state_dict(lora_weights, strict=False)
    # print("**********", len(unexpected_keys))
    value_model = QwenVLMValue(base, processor, 64, grpo = False)
    

    eval_loader = DataLoader(evalset, batch_size=4,  collate_fn=eval_collate_fn, shuffle=False) #collate_fn=data_collator,
    total_samples = 0
    total_squared_error = 0
    for batch in tqdm(eval_loader):
        scores, output = calc_utility_batch(value_model, processor, batch[1], batch[0])
        labels = batch[3]
        total_samples += len(labels)
        scores = np.array(scores)
        labels = np.array(labels)
        mse = np.sum((labels - scores) ** 2)
        total_squared_error += mse
    overall_mse = total_squared_error / total_samples
    print(f"Total squared error accumulated: {total_squared_error:.4f}")
    print(f"Total samples processed: {total_samples}")
    print(f"Overall MSE over batches: {overall_mse:.4f}")
    return None





#TODO s:
# model data type of bf16 or pf16 in case of lora
def main():
    global local_rank
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    random.seed(training_args.seed)
    torch.manual_seed(training_args.seed)
    torch.cuda.manual_seed_all(training_args.seed)
    
    # output_path = "/home/bahaduri/RL4VLM/outputs"
    # folder_path = output_path + f"/{trajNum}"
    # if not os.path.exists(folder_path):
    #     os.makedirs(folder_path)
    training_args.cuda = torch.cuda.is_available()
    if training_args.cuda and torch.cuda.is_available() and training_args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.set_num_threads(1)
    

    model_path = model_args.model_name_or_path

    processor = AutoProcessor.from_pretrained(model_path)
    print(model_path)
    if "Qwen2.5" in model_path:
        Qwen_model = Qwen2_5_VLForConditionalGeneration
    else:
        Qwen_model = Qwen2VLForConditionalGeneration
    
    base = Qwen_model.from_pretrained(model_path, torch_dtype=torch.float16, device_map="balanced")# low_cpu_mem_usage=True)
    # if 'mistral' in model_path.lower():
    #     base =  LlavaMistralForCausalLM.from_pretrained(model_path, cache_dir=cache_dir)
    # else:
    #     base = LlavaLlamaForCausalLM.from_pretrained(model_path, cache_dir=cache_dir)
    
    use_grad_ckpt = True
    if use_grad_ckpt:
        if hasattr(base, "enable_input_require_grads"):
            base.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            base.get_input_embeddings().register_forward_hook(make_inputs_require_grad)


    base.config.max_length = 1024
    print("Model max context length:{}".format(base.config.max_length))
    # base, tokenizer = init_pretrained_model(base, tokenizer, pretrain_mm_adapter = args.pretrain_mm_adapter)
    # image_processor = base.get_vision_tower().image_processor

    base_lora_config = LoraConfig(
            r=128,
            lora_alpha=256,
            target_modules=["q_proj", "v_proj", "k_proj", "mlp.0", "mlp.2"],#find_all_linear_names(base,args.train_vision),
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
    if training_args.lora_enable:
        base = get_peft_model(base, base_lora_config, adapter_name="policy")
        base.set_adapter("policy")
    
    data_module = make_supervised_data_module(processor=processor,
                                              data_args=data_args)
    trainer = VLMTrainer(model=base,
                    tokenizer=processor.tokenizer,
                    args=training_args,
                    **data_module)
    evaluate(base, data_module['eval_dataset'], processor)
    breakpoint()
    if list(Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

if __name__ == "__main__":
    main()



    