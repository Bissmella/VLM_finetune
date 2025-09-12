from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init
from a2c_ppo_acktr.llava_interface import llava_evaluate, llava_generate, qwen_generate, qwen_process, qwen_evaluate, qwen_evaluate_batch, qwen_generate_batch, qwen_calc_utility, qwen_calc_utility_batch
from .rl_utils import generate_fake_response
import torch.nn.init as init
import torchvision.transforms as T

from torch.nn import CrossEntropyLoss
from typing import List, Optional, Tuple, Union
from collections import deque
from transformers.modeling_outputs import ModelOutput

from torch.nn.utils.rnn import pad_sequence

import re
import json

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


@dataclass
class Qwen2VLCausalLMOutputWithPast(ModelOutput):
    """
    Base class for Qwen2VL causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None






class VLMValue(nn.Module):
    """
    actually the base is also used for generation!
    """
    def __init__(self, base):
        super(VLMValue, self).__init__()
        self.base = base
        # hard-code value head
        # self.value_head = nn.Linear(4096, 1, bias=True).to(base.device, dtype=torch.float16)
        self.value_head = nn.Sequential(
            nn.Linear(4096, 1024), # First layer
            nn.ReLU(), # Non-linearity
            nn.Linear(1024, 512), # Second layer
            nn.ReLU(), # Non-linearity
            nn.Linear(512, 1) # Output layer
            ).to(base.device, dtype=torch.float16) # Move to specified device with dtype

    def forward(self,  input_ids, image_tensor):
        if image_tensor.size(0) != 1:
            input_ids = input_ids.broadcast_to(image_tensor.size(0), input_ids.size(-1))

        image_tensor = image_tensor.to(self.base.device, dtype = self.base.dtype)
        _, _, _, _, inputs_embeds, _ = self.base.prepare_inputs_labels_for_multimodal(input_ids.to(self.base.device), None, None, None, None, image_tensor)
        inputs_embeds = inputs_embeds.to(self.base.device, dtype = self.base.dtype)
        assert inputs_embeds.shape[1] > 256
        outputs = self.base(
            inputs_embeds = inputs_embeds,
            output_hidden_states=True)
        hidden_states = outputs.hidden_states
        values = self.value_head(hidden_states[-1][:, -1])
        return values


class VLMPolicy(nn.Module):
    def __init__(self, tokenizer,
                image_processor,
                value_model,
                args,
                INPUT_IDS,
                projection_f,
                base_kwargs=None):
        """
        projection_f: the postprocessing function to parse text action
        """
        super(VLMPolicy, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.value_model = value_model
        self.base = value_model.base
        self.INPUT_IDS = INPUT_IDS
        self.projection_f = projection_f

    def process_obs(self, obs):
        #process the observation with the image processor
        processed_images = obs
        return self.image_processor.preprocess(processed_images, return_tensors='pt')['pixel_values'].to(dtype=self.base.dtype)

    def act(self, inputs, deterministic=False, INPUT_IDS=None):
        image_tensor = self.process_obs(inputs)
        if INPUT_IDS is None:
            INPUT_IDS = self.INPUT_IDS
        value, output_ids, text_action, action_log_prob, action_tokens_log_prob = llava_generate(value_model = self.value_model,
                                                    tokenizer = self.tokenizer,
                                                    input_ids = INPUT_IDS,
                                                    image_tensor = image_tensor,
                                                    args = self.args)
        action = self.projection_f(text_action)
        return value, output_ids, action, action_log_prob, action_tokens_log_prob

    def get_value(self, inputs, INPUT_IDS=None):
        if INPUT_IDS is None:
            INPUT_IDS = self.INPUT_IDS
        image_tensor = self.process_obs(inputs)
        return self.value_model(input_ids = INPUT_IDS, image_tensor = image_tensor)

    def evaluate_actions(self, inputs, output_ids, INPUT_IDS=None):
        image_tensor = self.process_obs(inputs)
        if INPUT_IDS is None:
            INPUT_IDS = self.INPUT_IDS
        value, action_log_prob, _ = llava_evaluate(value_model = self.value_model,
                                        input_ids = INPUT_IDS,
                                        output_ids = output_ids,
                                        image_tensor = image_tensor,
                                        temperature = self.args.temperature,
                                        thought_prob_coef = self.args.thought_prob_coef)
        return value, action_log_prob


class QwenVLMValue(nn.Module):
    """
    extends base as value model by attaching a value head.
    """
    def __init__(self, base, processor, hidden_dim=1536, grpo=False):
        # First layer  #the hidden states of the qwen has 1536 dims
        super(QwenVLMValue, self).__init__()
        self.base = base
        self.processor = processor
        
        if not grpo:
            self.value_head = nn.Sequential(
                nn.Linear(hidden_dim, 1024), 
                nn.ReLU(), # Non-linearity
                nn.Linear(1024, 512), # Second layer
                nn.ReLU(), # Non-linearity
                nn.Linear(512, 1) # Output layer
                ).to(base.device, dtype=torch.float16) # Move to specified device with dtype
        else:
            self.value_head = None
    
    def get_value(self, hidden_states):
        """
        different from policy get_value, this is getting value based on computed hidden states
        """
        if self.value_head is not None:
            return self.value_head(hidden_states)
        else:
            bs = hidden_states.shape[0]
            out= torch.zeros((bs, 1), device=self.base.device)
            return out

    def forward(self, text = None, image= None, inputs = None):
        if text is not None and image is not None:
            return self.forward_val(text, image)
        else:
            return self.forward_tmp(**inputs)
        
    def forward_val(self,  text, image):
        """
        gets value of an action.
        
        text: string - textual prompt
        image: tensor - image observation from environment

        returns:
        if value head is present:
            values: tensor shape[1,1] - the value of action based on value head
        else:
            out: tensor shape[1,1] - a 0 value
        """
        if self.value_head is not None:
            input = qwen_process(self.processor, text, image)
            input = input.to(self.base.device)
            outputs = self.base(
                **input,
                output_hidden_states=True)
            hidden_states = outputs.hidden_states
            values = self.value_head(hidden_states[-1][:, -1])
            return values
        else:
            out= torch.zeros((1, 1), device=self.base.device)
            return out
    
    def forward_tmp(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        loss_weights: Optional[torch.LongTensor] =None,
    ) -> Union[Tuple, Qwen2VLCausalLMOutputWithPast]:
        r"""
        This is used for temporal predictability generation. Predicts the next action given the previous sequence of actions.

        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        pixel_values_videos (`torch.FloatTensor` of shape `(seq_length, num_channels * temporal_size * image_size * image_size)):
            The tensors corresponding to the input videos. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`Qwen2VLImageProcessor.__call__`] for details. [`Qwen2VLProcessor`] uses
            [`Qwen2VLImageProcessor`] for processing videos.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        >>> model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.base.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.base.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.base.config.use_return_dict

        outputs = self.base(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        
        
        logits = outputs.logits

        loss = None
        if labels is not None:
            if loss_weights is not None:
                loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
                loss_raw = loss * loss_weights.view(-1)
                loss = torch.mean(loss_raw)
            else:
                loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
                loss_raw=None

        return Qwen2VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        ), loss_raw



class QwenVLMPolicy(nn.Module):
    def __init__(self,
                processor,
                value_model,
                args,
                INPUT_IDS,
                projection_f,
                base_kwargs=None):
        """
        projection_f: the postprocessing function to parse text action
        """
        super(QwenVLMPolicy, self).__init__()
        self.args = args
        self.processor = processor
        self.value_model = value_model
        self.base = value_model.base
        self.INPUT_IDS = INPUT_IDS
        self.projection_f = projection_f

    def process_obs(self, obs):
        #process the observation with the image processor
        processed_images = obs
        return self.image_processor.preprocess(processed_images, return_tensors='pt')['pixel_values'].to(dtype=self.base.dtype)

    def act(self, image, deterministic=False, text=None):
        """
        creates prompt out of the image and text.
        In case of self.args.action_sampling: uses score policy
            If deterministic:
                takes max over score policy
            else:
                sample from score policy
        If self.args.action_sampling is False: uses standard act policy
        """
        if self.args.action_sampling:
            text_action, output_ids, output_ids_trimmed = qwen_generate(value_model = self.value_model,
                                                        processor = self.processor,
                                                        text = text,
                                                        image=image,
                                                        args = self.args)
            action, random_mask, command, log_prob, thts_list = self.projection_f(text_action, action_sampling = self.args.action_sampling, deterministic= deterministic)
            
            fake_response = generate_fake_response( text_action, command, thts_list) #new correctly formatted response based on the command
        
            f_response_encoded = self.processor.tokenizer(fake_response, padding=True, return_tensors="pt")["input_ids"]
            
            padded_output_ids_trimmed = pad_sequence(f_response_encoded, batch_first=True, padding_value=0)
            padded_output_ids = torch.full((output_ids.size(0), 2*self.args.max_new_tokens), 151643, dtype=output_ids.dtype, device = output_ids.device) #151643 is pad token in qwen2vl #TODO hardcoded
            padded_output_ids[:, :padded_output_ids_trimmed.size(1)] = padded_output_ids_trimmed
            if self.args.grpo: #in the case of grpo no need to compute value, and log_probs as the response will be regenerated in generate_input
                values = qwen_evaluate(self.value_model, padded_output_ids, self.args.temperature, self.args.thought_prob_coef, self.processor, text=self.INPUT_IDS, image=image, value_only= True)
                sum_log_probs, action_tokens_log_prob = torch.tensor([0]), torch.tensor([0])
            else:
                with torch.no_grad():
                    values, sum_log_probs, action_tokens_log_prob = qwen_evaluate(self.value_model, padded_output_ids, self.args.temperature, self.args.thought_prob_coef, self.processor, text=self.INPUT_IDS, image=image  )
            
            return values, padded_output_ids, action, random_mask, command, log_prob, action_tokens_log_prob, fake_response
        

        else: #using standard policy
            value, output_ids, text_action, action_log_prob, action_tokens_log_prob = qwen_generate(value_model = self.value_model,
                                                        processor = self.processor,
                                                        text = text,
                                                        image=image,
                                                        args = self.args)
            action, random_mask, command = self.projection_f(text_action, action_sampling = self.args.action_sampling)
            # if random_mask == 1:  #in the case of grpo the output_ids will be regenerated anyway #TODO to be cleaned
                
            #     fake_response = generate_fake_response( text_action, command)
            #     f_response_encoded = self.processor.tokenizer(fake_response, padding=True, return_tensors="pt")["input_ids"]
            #     padded_output_ids_trimmed = pad_sequence(f_response_encoded, batch_first=True, padding_value=0)
            #     padded_output_ids = torch.full((output_ids.size(0), 2*self.args.max_new_tokens), 151643, dtype=output_ids.dtype, device = output_ids.device) #151643 is pad token in qwen2vl #TODO hardcoded
            #     padded_output_ids[:, :padded_output_ids_trimmed.size(1)] = padded_output_ids_trimmed
            #     output_ids = padded_output_ids
            #     with torch.no_grad():
            #         values, action_log_prob, action_tokens_log_prob = qwen_evaluate(self.value_model, padded_output_ids, self.args.temperature, self.args.thought_prob_coef, self.processor, text=self.INPUT_IDS, image=image  )
            return value, output_ids, action, random_mask, command, action_log_prob, action_tokens_log_prob
        
    def act_batch(self, inputs, text=None, group=4):
        """
        same as act but in batch mode supporting group generation (multiple generation for each prompt similar to GRPO).
        inputs: List[tensors] - list of image
        text: string - the standard prompt 
        """
        bs = inputs.shape[0]
        images = []
        for img in inputs:
            image_tensor = img.squeeze(0).permute(2,0,1).float()
            if image_tensor.max() <= 1.0:
                image_tensor = (image_tensor * 255).byte()
            to_pil = T.ToPILImage()
            image = to_pil(image_tensor)
            for i in range(group):
                images.append(image)
        if text is None:
            INPUT_IDS = [self.INPUT_IDS for _ in range(len(images))]
        else:
            INPUT_IDS = [text for _ in range(group)]
        
        output_text, output_ids, action_log_probs = qwen_generate_batch(self.value_model, self.processor, INPUT_IDS, images, self.args)
        return output_text, output_ids, action_log_probs

    def get_value(self, image, text=None):
        if text is None:
            text = self.INPUT_IDS
        return self.value_model(text = text, image = image)

    def evaluate_actions(self, inputs, output_ids, INPUT_IDS=None):
        """
        Calculates value of pre-generated action

        inputs: tensor - image
        INPUT_IDS: string - the text prompt
        output_ids:  tensor - ids of pre-generated response
        """
        assert inputs.shape[0] == 1, "multip image in action evaluation!"
        image_tensor = inputs.squeeze(0).permute(2,0,1).float()
        
        if image_tensor.max() <= 1.0:
            image_tensor = (image_tensor * 255).byte()
        to_pil = T.ToPILImage()
        image = to_pil(image_tensor)
        
        if INPUT_IDS is None:
            INPUT_IDS = self.INPUT_IDS
        output_ids = output_ids.to(self.base.device)
        value, action_log_prob, _ = qwen_evaluate(value_model = self.value_model,
                                        output_ids = output_ids,
                                        temperature = self.args.temperature,
                                        thought_prob_coef = self.args.thought_prob_coef,
                                        processor=self.processor,
                                        text = INPUT_IDS,
                                        image = image,)
        return value, action_log_prob
    
    def evaluate_actions_batch(self, inputs, output_ids, INPUT_IDS=None):
        """
        same as above but in batch mode. The result is not consistent with evaluate_actions!!! #TODO

        inputs: List[tensors] - list of input images
        INPUT_IDS: string - the standard text prompt (same prompt for all images will be used)
        output_ids: tensor - pre-generated response

        returns:
            value: tensor - value of the action
            action_log_prob: tensor - log probability of the action

        """
        bs = output_ids.shape[0]
        if inputs.shape[0] != output_ids.shape[0]:
            inputs = inputs.repeat(bs, 1, 1, 1)
        images = []
        
        for img in inputs:
            image_tensor = img.squeeze(0).permute(2,0,1).float()
            if image_tensor.max() <= 1.0:
                image_tensor = (image_tensor * 255).byte()
            to_pil = T.ToPILImage()
            image = to_pil(image_tensor)
            images.append(image)
        if INPUT_IDS is None:
            INPUT_IDS = [self.INPUT_IDS for _ in range(bs)]
        
        output_ids = output_ids.to(self.base.device)
        
        value, action_log_prob, _ = qwen_evaluate_batch(value_model= self.value_model,
                                                 output_ids = output_ids,
                                                 temperature = self.args.temperature,
                                                 thought_prob_coef = self.args.thought_prob_coef,
                                                 processor = self.processor,
                                                 text= INPUT_IDS,
                                                 image = images,
                                                 grpo = False)
        return value, action_log_prob

    def calc_utility(self, images, INPUT_IDS):
        """
        calculates utility of an action based on before and after image in the environment using the guidance in prompt (INPUT_IDS).

        images: list[tensors] - list of before image and after image (based on the action)
        INPUT_IDS: string - textual prompt describing the action taken and scoring scale

        returns:
        action_scores: int - the utility score given to the action if output was parsed else -1
        outputs: string - the textual output of the model
        
        """
        outputs = qwen_calc_utility(value_model = self.value_model,
                                                        processor = self.processor,
                                                        text = INPUT_IDS,
                                                        images=images,
                                                        args = self.args)
        
        string = outputs[0]
        string = string.lower()
        try:
            match = re.search(r"```(?:json)?\n(.*?)\n```", string, re.DOTALL)
            if match:
                string = match.group(1)
            else:
                string = string.strip()
            response_json = json.loads(string)
            action_scores = response_json["score"]
        except:
            action_scores = -1
        return action_scores, outputs[0]
    
    def calc_utility_batch(self, images, INPUT_IDS):
        """
        Same as above but in batch mode.

        returns:
        same as above
        bad_util: int - number of bad and unparseable outputs
        """
        
        outputs = qwen_calc_utility_batch(value_model = self.value_model,
                                                        processor = self.processor,
                                                        text = INPUT_IDS,
                                                        images=images,
                                                        args = self.args)
        action_scores = []
        bad_util=0
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
                bad_util += 1
            action_scores.append(action_score)
        return action_scores, outputs, bad_util



class QwenTempPredictor(nn.Module):
    def __init__(self,
                processor,
                base,
                args=None,
                INPUT_IDS= None,
                base_kwargs=None):
        super(QwenTempPredictor, self).__init__()
        self.base = base
        self.processor = processor
        self.args = args
        self.config = base.config

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        loss_weights: Optional[torch.LongTensor] =None,
    ) -> Union[Tuple, Qwen2VLCausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        pixel_values_videos (`torch.FloatTensor` of shape `(seq_length, num_channels * temporal_size * image_size * image_size)):
            The tensors corresponding to the input videos. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`Qwen2VLImageProcessor.__call__`] for details. [`Qwen2VLProcessor`] uses
            [`Qwen2VLImageProcessor`] for processing videos.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        >>> model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.base(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        
        #hidden_states = outputs.hidden_states#outputs[0]
        logits = outputs.logits#outputs[0]#self.base.lm_head(hidden_states)

        loss = None
        if labels is not None:
            if loss_weights is not None:
                loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
                loss_raw = loss * loss_weights.view(-1)
                loss = torch.mean(loss_raw)
            else:
                loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
                loss_raw=None

        return Qwen2VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        ), loss_raw