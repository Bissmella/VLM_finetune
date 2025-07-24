import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from transformers import DataCollatorForSeq2Seq
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import warnings
import numpy as np
import gc
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from typing import List, Optional, Tuple, Union
from collections import deque
from a2c_ppo_acktr.llava_interface import qwen_process, qwen_batch_process, format_data_sft
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLProcessor
from peft import get_peft_model_state_dict
# from datasets import Dataset
import accelerate

ALF_ACTION_LIST=["pass", "goto", "pick", "put", "open", "close", "toggle", "heat", "clean", "cool", "slice", "inventory", "examine", "look"]



def dict_mean(dict_list):
    mean_dict = {}
    if len(dict_list) > 0:
        for key in dict_list[0].keys():
            if "min" in key:
                mean_dict[key] = min(d[key] for d in dict_list)
            elif "max" in key:
                mean_dict[key] = max(d[key] for d in dict_list)
            else:
                mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict



class TrajectoryDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings["input_ids"])

class TrajectoryDatasetList(Dataset):
    def __init__(self, encodings, rnd_masks, bad_masks, status):
        self.encodings = encodings
        self.rnd_masks = rnd_masks
        self.bad_masks = bad_masks
        self.status = status

    def __getitem__(self, idx):
        try:
            return self.encodings[idx] , self.rnd_masks[idx], self.bad_masks[idx], self.status[idx]  # simply return the dict at index `idx`
        except:
            breakpoint()
    def __len__(self):
        return len(self.encodings)


class CustomCollator:
    def __init__(self, tokenizer, model):
        self.base_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    def __call__(self, batch):
        # Separate the weights before calling the base collator
        weights = [torch.tensor(example["weights"], dtype=torch.float) for example in batch]

        # Remove weights from examples to avoid issues with the base collator
        for example in batch:
            del example["weights"]

        # Use Huggingface's default collator to pad everything else
        batch_out = self.base_collator(batch)

        # Pad weights to match the padded labels
        padded_weights = pad_sequence(weights, batch_first=True, padding_value=1.0)  # default neutral weight

        batch_out["weights"] = padded_weights

        return batch_out


def freeze_all_except_adapter(model, active_adapter_name):
    for name, param in model.named_parameters():
        if f"lora_A.{active_adapter_name}" in name or f"lora_B.{active_adapter_name}" in name:
            param.requires_grad = True
        elif "lora_A." in name or "lora_B." in name:
            param.requires_grad = False



class Temp_predictor():
    def __init__(self, model, processor, optimizer, accelerator, epochs=4, temp_model_lr=3e-4, buff_size=160):

        """
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        model = T5Model.from_pretrained("t5-small")
        """
        self.accelerator = accelerator#accelerate.Accelerator()
        self.act_sep_token = '||'  #special action seperator token
        self.processor = processor#T5TokenizerFast.from_pretrained("t5-small", extra_special_tokens={"act_token": self.act_sep_token})
        # self.model = model
        # #breakpoint()
        # self.model.base.set_adapter('adversery')
        #self.model = T5ForConditionalGeneration.from_pretrained("t5-small")
        # freeze_all_except_adapter(self.model.base, "adversery")
        # trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        # adapter_params = get_peft_model_state_dict(self.model.base, adapter_name="adversery")
        self.temp_model_optimizer = optimizer#torch.optim.Adam(trainable_params, lr= temp_model_lr)#self.model.parameters(), lr = temp_model_lr)
        self.device = model.base.device
        #self.accelerator = accelerator
        self.epochs = epochs
        self.model = model
        # self.processor.tokenizer.add_special_tokens({'additional_special_tokens': [self.act_sep_token]})  #action seperator token  additional_special_tokens   act_token
        # self.model.base.resize_token_embeddings(len(self.processor.tokenizer))
        self.act_sep_token_id = self.processor.tokenizer.encode(" " + self.act_sep_token)  #space is because one space coming before it changes the token id
        #self.temp_model_optimizer = self.accelerator.prepare(self.temp_model_optimizer)
        
        #self.model.to(self.device)
        #adapter_state_dict = get_peft_model_state_dict(self.model.base)
        #breakpoint()
        #self.model, self.temp_model_optimizer = self.accelerator.prepare(self.model, self.temp_model_optimizer)
        #trajectories buffer
        self.input_seq_buffer = deque(maxlen=buff_size)
        self.target_seq_buffer = deque(maxlen=buff_size)

        #trajectories buffer for multimodal that will store dictionaries of trajectory
        self.trajs_buffer = deque(maxlen=buff_size)
        self.trajs_rndMask_buffer = deque(maxlen=buff_size)
        self.trajs_badMask_buffer = deque(maxlen=buff_size)
        self.status_buffer = deque(maxlen=buff_size)
        self.spec_gen_token = "%generate%"
        # for name, param in self.model.named_parameters():
        #     if hasattr(param, 'ds_id'):
        #         print(f"{name} is managed by deepspeed!")
        # breakpoint()
        # print("here")
        #adapter_state_dict = get_peft_model_state
        #_dict(self.model.base)
        #breakpoint()
        
        


    def preprocess_trajectories(self, buffer_goals, buffer_trajLen, buffer_actions, terminals):
        input_seq = []
        target_seq = []
        counter =0
        for i, len in enumerate(buffer_trajLen):
            acts = buffer_actions[counter: counter + len]
            counter += len
            if terminals[i]:
                self.input_seq_buffer.append(buffer_goals[i])
                flattened = f" {self.act_sep_token} ".join(acts)
                self.target_seq_buffer.append(flattened)
        # input_seq= None
        # target_seq= None
        return list(self.input_seq_buffer), list(self.target_seq_buffer)
    

    def preprocess_mm_trajectories(self, buffer_goals, buffer_actions, buffer_obs, buffer_trajLen, random_masks, bad_masks, status=None):
        #TODO add terminals to add trajectory only if it is terminated for updating the model
        trajs = []
        counter = 0
        for i, seg_len in enumerate(buffer_trajLen):
            traj = {}
            acts = buffer_actions[counter: counter + seg_len]   
            flattened_acts = f" {self.act_sep_token} ".join(acts)
            flattened_acts = f" {self.spec_gen_token} " + flattened_acts
            counter += seg_len
            traj["image"] =buffer_obs[i]
            traj["query"] = buffer_goals[i]
            traj["label"] = flattened_acts
            #traj["rnd_mask"] = random_masks[i]
            trajs.append(traj)
            self.trajs_buffer.append(traj)
            self.trajs_rndMask_buffer.append(random_masks[i])
            self.trajs_badMask_buffer.append(bad_masks[i])
            self.status_buffer.append(status[i])
            
            
        return trajs##input_seq, target_seq, obs_seq
    
    def process_mm(self, trajectories):

        output = qwen_batch_process(self.processor, trajectories["texts"], trajectories['images'])
        output_target = qwen_batch_process(self.processor, trajectories["target"])
        breakpoint() #TODO   check and extract required things from output
        return output
    
    def collate_fn(self, input):
        """
        examples has a length of batch-size that is not fixed and can vary
        each consist of tuple: first is chat, second is random-mask
        """
        # Get the texts and images, and apply the chat template
        examples = [example[0] for example in input]
        random_masks = [example[1] for example in input]
        bad_masks = [example[2] for example in input]
        status = [example[3] for example in input]
        
        status = torch.tensor(status, dtype=torch.long)
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
        labels, batch_slices, random_coeffs = self.assign_random_mask_weights(labels, random_masks, bad_masks)
        pattern = torch.tensor([1018, 19366, 4], device=labels.device)   #that is fixed token id of a special token  " %generate% "
        match = (labels.unfold(1, len(pattern), 1) == pattern).all(-1)
        first_match = torch.where(match.any(1), match.float().argmax(1), torch.full((labels.size(0),), labels.size(1), device=labels.device))
        first_match = torch.where(match.any(1), match.float().argmax(1), torch.full((labels.size(0),), -len(pattern), device=labels.device))
        mask = torch.arange(labels.size(1), device=labels.device).unsqueeze(0) < (first_match + len(pattern)).unsqueeze(1)
        # Mask image token IDs in the labels
        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100  # Mask image token IDs in labels
        labels[mask] = -100
        
        batch["labels"] = labels  # Add labels to the batch
        batch["labels"] = batch["labels"].to(self.device)
        batch["input_ids"] = batch['input_ids'].to(self.device)
        batch['attention_mask'] = batch['attention_mask'].to(self.device)
        batch['pixel_values'] = batch['pixel_values'].to(self.device)
        batch['image_grid_thw'] = batch['image_grid_thw'].to(self.device)
        return batch, batch_slices, random_coeffs, status
    
    def preprocess_data(self, examples):
        model_inputs = self.tokenizer(examples["input"], truncation=True, padding=False)
        labels = self.tokenizer(examples["target"], return_offsets_mapping=True, truncation=True, padding=False)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs, labels["offset_mapping"]

    def update_model(self, buffer_goals, buffer_trajLen, buffer_actions, terminals):
        self.model.train()
        input_seq, target_seq = self.preprocess_trajectories(buffer_goals, buffer_trajLen, buffer_actions, terminals) #train_data
        model_inputs = self.tokenizer(
            input_seq, #[inp for inp, _ in train_data],
            padding=False, #"max_length",
            truncation=True,
            #max_length=64,
            return_tensors=None, #"pt"
        )
        labels = self.tokenizer(
            target_seq, #[tgt for _, tgt in train_data],
            return_offsets_mapping=True,
            padding=False, #"max_length",
            truncation=True,
            #max_length=64,
            return_tensors=None, #"pt"
        )
        token_offsets = labels["offset_mapping"]
        model_inputs["labels"] = labels["input_ids"]
        model_inputs = self.assign_token_weights2(target_seq, model_inputs, token_offsets)
        dataset = TrajectoryDataset(model_inputs)
        data_collator = CustomCollator(tokenizer=self.tokenizer, model=self.model)
        train_loader = DataLoader(dataset, batch_size=64, collate_fn=data_collator, shuffle=True)
        #TODO set the max new tokens of the model to max_tokens of the policy LLM * max_steps of the policy
        #train_loader = self.accelerator.prepare(train_loader)
        info = {}
        info_list = []
        for _ in tqdm(range(self.epochs)):
            self.temp_model_optimizer.zero_grad()
            epoch_loss= 0
            correct = 0
            total = 0
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                input_attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                loss_weights = batch['weights'].to(self.device)
                #labels_attention_mask = label['attention_mask'].to(self.device)

                outputs = self.model(input_ids=input_ids,
                                    attention_mask=input_attention_mask,
                                    encoder_outputs=None,
                                    labels= labels,
                                    loss_weights = loss_weights,
                                    # decoder_input_ids=labels_input_ids,
                                    # decoder_attention_mask=labels_attention_mask,
                                    )
                loss, logits = outputs[0].loss, outputs[0].logits  #TODO  3 is placeholder for the logits #outputs.loss, outputs.logits
                #self.accelerator.backward(loss)
                #self.accelerator.backward(loss)
                loss.backward()
                epoch_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)
                mask = labels != -100
                correct += ((preds == labels) & mask).sum().item()
                total += mask.sum().item()
            avg_epoch_loss = epoch_loss / len(train_loader)
            train_accuracy = correct / total
            info_list.append({"temp_predictor.loss": avg_epoch_loss, "temp_predictor.acc": train_accuracy})
            #self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.temp_model_optimizer.step()
            torch.cuda.empty_cache()
            gc.collect()

        info.update(dict_mean(info_list))
        return info
    

    def update_mm_model(self):
        trajs = self.trajs_buffer
        random_masks = self.trajs_rndMask_buffer
        bad_masks = self.trajs_badMask_buffer
        trajs = [format_data_sft(traj) for traj in trajs]
        status = self.status_buffer
        
        trajs_dataset = TrajectoryDatasetList(trajs, random_masks, bad_masks, status)

        train_loader = DataLoader(trajs_dataset, batch_size=4, collate_fn=self.collate_fn, shuffle=True)
        train_loader = self.accelerator.prepare(train_loader)
        info = {}
        info_list = []
        #freeze_all_except_adapter(self.model.base, "adversery")
        self.model.base.set_adapter('adversery')
        self.model.train()
        for _ in tqdm(range(self.epochs)):
            self.temp_model_optimizer.zero_grad()
            epoch_loss= 0
            correct = 0
            total = 0
            for batch, _, _, _ in train_loader:
                
                labels = batch['labels']
                outputs = self.model(inputs = batch,
                                        #output_hidden_states = True,
                                        # decoder_input_ids=labels_input_ids,
                                        # decoder_attention_mask=labels_attention_mask,
                                        #TODO loss_weights
                                        )
                loss = outputs[0].loss.view(batch['input_ids'].shape)
                loss = torch.mean(loss)
                logits = outputs[0].logits  #TODO  3 is placeholder for the logits #outputs.loss, outputs.logits
                #self.accelerator.backward(loss)
                #breakpoint()
                #loss.backward()
                self.accelerator.backward(loss)
                epoch_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)
                mask = labels != -100
                correct += ((preds == labels) & mask).sum().item()
                total += mask.sum().item()
            avg_epoch_loss = epoch_loss / len(train_loader)
            train_accuracy = correct / total
            info_list.append({"temp_predictor.loss": avg_epoch_loss, "temp_predictor.acc": train_accuracy})
            #self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.temp_model_optimizer.step()
            torch.cuda.empty_cache()
            gc.collect()

        info.update(dict_mean(info_list))
        #freeze_all_except_adapter(self.model.base, "policy")
        self.model.base.set_adapter('policy')
        return info
            


    def eval_model(self, sequence):
        """
        To be used for both validation of the temporal predictor model and also for novelty score calculation.
        """
        self.model.eval()
        eval_data = self.preprocess_trajectories(sequence)
        model_inputs = self.tokenizer(
            [inp for inp, _ in eval_data],
            padding=False, #"max_length",
            truncation=True,
            #max_length=64,
            return_tensors=None, #"pt"
        )
        labels = self.tokenizer(
            [tgt for _, tgt in eval_data],
            return_offsets_mapping=True,
            padding=False, #"max_length",
            truncation=True,
            #max_length=64,
            return_tensors=None, #"pt"
        )
        token_offsets = labels["offset_mapping"]
        model_inputs["labels"] = labels["input_ids"]
        model_inputs = self.assign_token_weights2([tgt for _, tgt in eval_data], model_inputs, token_offsets)
        eval_dataset = TrajectoryDataset(model_inputs)
        data_collator = CustomCollator(tokenizer=self.tokenizer, model=self.model)
        eval_loader = DataLoader(eval_dataset, batch_size=64, collate_fn=data_collator, shuffle=False)
        
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in eval_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    input_attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    loss_weights = batch['weights'].to(self.device)
                    #labels_attention_mask = label['attention_mask'].to(self.device)

                    outputs = self.model(input_ids=input_ids,
                                        attention_mask=input_attention_mask,
                                        encoder_outputs=None,
                                        # decoder_input_ids=labels_input_ids,
                                        # decoder_attention_mask=labels_attention_mask,
                                        )
                    loss, logits = outputs[0].loss, outputs[0].logits
                    val_loss += loss.item()
                    preds = torch.argmax(logits, dim=-1)
                    mask = labels != -100
                    correct += ((preds == labels) & mask).sum().item()
                    total += mask.sum().item()
        avg_val_loss = val_loss/ len(eval_loader)
        val_accuracy = correct/total
        
        #
        return loss   #higher = more novel
    
    def compute_novelty(self, input, target, image=None):
        """
        computes novetly for one trajectory
        input: a string, mainly the trajectory's goal
        target: list of strings, mainly the trajectory's actions
        """
        freeze_all_except_adapter(self.model.base, "adversery")
        self.model.base.set_adapter('adversery')
        self.model.eval()
        target = f" {self.act_sep_token} ".join(target)
        model_inputs = self.tokenizer(
            [input],
            padding=False, #"max_length",
            truncation=True,
            #max_length=64,
            return_tensors=None, #"pt"
        )
        labels_ = self.tokenizer(
            [target],
            return_offsets_mapping=True,
            padding=False, #"max_length",
            truncation=True,
            #max_length=64,
            return_tensors=None, #"pt"
        )
        
        #creating slices of actions
        #TODO check the labels might need an indexing like labels[0]
        act_sep_indices = [i for i, x in enumerate(labels_["input_ids"][0]) if x == self.act_sep_token_id]
        start_indices = [0] + [i + 1 for i in act_sep_indices]
        end_indices = act_sep_indices + [len(labels_["input_ids"][0])-1]
        # Now pair them into slices
        action_slices = [slice(start, end) for start, end in zip(start_indices, end_indices)]

        token_offsets = labels_["offset_mapping"]
        model_inputs["labels"] = labels_["input_ids"]
        model_inputs = self.assign_token_weights2([target], model_inputs, token_offsets)
        #TODO put a breakpoint here and check if no index [0] needed after the model_inputs
        
        input_ids = torch.tensor(model_inputs["input_ids"]).to(self.device)
        input_attention_mask = torch.tensor(model_inputs['attention_mask']).to(self.device)
        loss_weights = torch.tensor(model_inputs['weights']).to(self.device)
        labels = torch.tensor(model_inputs['labels']).to(self.device)
        outputs = self.model(input_ids=input_ids,
                                        attention_mask=input_attention_mask,
                                        encoder_outputs=None,
                                        labels = labels,
                                        loss_weights=loss_weights,
                                        # decoder_input_ids=labels_input_ids,
                                        # decoder_attention_mask=labels_attention_mask,
                                        )
        
        loss_raw = outputs[1]
        
        loss_sliced = [loss_raw[s] for s in action_slices]
        action_loss_normalized = np.array([s.mean().item() for s in loss_sliced])
        epsilon = 1e-8
        final_loss = (action_loss_normalized - np.min(action_loss_normalized))/(np.max(action_loss_normalized) - np.min(action_loss_normalized) + epsilon)

        freeze_all_except_adapter(self.model.base, "policy")
        self.model.base.set_adapter('policy')

        return final_loss
    

    def compute_novelty_batch(self, inputs, targets, images, traj_lens,  random_masks=None, bad_masks=None, status=None):
        """
        computes novelty of batch of trajectories involving multimodality.
        input:
        target:
        image:
        random_masks:
        bad_masks: 
        """
        self.model.eval()
        trajs = self.preprocess_mm_trajectories(inputs, targets, images, traj_lens, random_masks, bad_masks, status) #TODO udpate the preprocess for bad_masks, collate_fn as well
        try:
            trajs = [format_data_sft(traj) for traj in trajs]
        except:
            breakpoint()
        trajs_dataset = TrajectoryDatasetList(trajs, random_masks, bad_masks, status)

        data_loader = DataLoader(trajs_dataset, batch_size=8, collate_fn=self.collate_fn, shuffle=False)
        losses = []
        with torch.no_grad():
            for batch, batch_slices, random_coeffs, status in data_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    input_attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)


                    outputs = self.model(inputs = batch,
                                        
                                        )
                    loss = outputs[0].loss.view(batch['input_ids'].shape)
                    
                    #batch_slices are slices of proper actions and random ones.
                    action_slices = [
                                [slice(start, end) for start, end in zip(s[:, 0].tolist(), s[:, 1].tolist())]
                                if s is not None else None
                                for s in batch_slices
                            ]

                    #loss sliced to each proper action
                    loss_sliced = [
                        [loss_raw[s] for s in slices] if slices is not None
                        else [loss_raw.new_tensor(0)]  # better device/dtype safety
                        for loss_raw, slices in zip(loss, action_slices)
                    ]

                    action_loss_aggregated = [
                        torch.tensor([s.sum().item() for s in sliced], device=sliced[0].device)
                        for sliced in loss_sliced
                    ]
                    
                    #setting the loss for succcessful trajs to 0
                    
                    action_loss_aggregated = [loss * random_coeff.to(loss.device) for loss, random_coeff in zip(action_loss_aggregated, random_coeffs)]
                    status = 1- status  #converting status -->  1: failed,  0: success in order to target only failed trajectories
                    action_loss_final = [loss * stat for loss, stat in zip(action_loss_aggregated, status)]
                    losses.extend(action_loss_final)
                    # #seperators = [[slice[0, 0].item() + slice[:, 1].tolist()] for slice in batch_slices]
                    # slices = [torch.cat([torch.arange(*slicee) for slicee in inverse_slices]) for inverse_slices in batch_slices]
                    # losses.append(loss)
        
        losses = torch.cat(losses, dim=0)
        epsilon = 1e-8
        final_loss = (losses - torch.min(losses))/(torch.max(losses) - torch.min(losses) + epsilon)
        return final_loss

    
    def assign_random_mask_weights(self, labels, random_masks, bad_masks):
        """
        Makes the random_masks match the tokens.
        texts: list(str), list of action sequences
        model_inputs: dictionary including tokenized action sequences
        random_masks: list or tensor of shape (len(texts), len(actions sequence))
        """
        pattern = torch.tensor([1018, 19366, 4], device=labels.device)   #that is fixed token id of a special token  " %generate% "
        matched = (labels.unfold(1, len(pattern), 1) == pattern).all(-1)
        #first_match = torch.where(match.any(1), match.float().argmax(1), torch.full((labels.size(0),), labels.size(1), device=labels.device))
        first_match = torch.where(matched.any(1), matched.float().argmax(1), torch.full((labels.size(0),), -len(pattern), device=labels.device))
        first_match += 2
        
        seperator_mask = (labels == self.act_sep_token_id[0])   #finding where in the labels are action seperators " ||"
        try:
            indices_per_row = [row.nonzero(as_tuple=True)[0] for row in seperator_mask]  # indices where the action seperators exists
        except:
            breakpoint()
        batch_slices = []
        random_coeffs = []
        
        for indices, first, label, random_mask, bad_mask in zip (indices_per_row, first_match, labels, random_masks, bad_masks):
            indices_mask = (indices > first)  #the actual action seperators is after the first (%generate% token)
            indices = indices[indices_mask]   #subetting based on above flag
            
            len_label = torch.Tensor([len(label)], device=label.device).int() # dtype=torch.long,
            first = first.unsqueeze(0)
            indices = torch.cat((first, indices, len_label), dim=0)    #first is the start, then intermidary indices and the final which is length of labels
            
            slices = indices.unfold(0, 2, 1)  #turning indices to slices
            
            # random_mask = random_mask.unsqueeze(1)
            # random_mask = random_mask.expand(slices.shape)
            mask = random_mask == 1    #mask for random actions to not be considered into calculation
            slices_copy = slices.clone()
            slices = slices[mask]       #random actions
            random_coeff = torch.ones(slices_copy.shape[0], device = label.device)  #randomness coefficents, ones means proper actions
            random_coeff[mask] = 0   #setting coefficient for random actions to 0 
            inverse_slices = slices_copy   #actions taken by both policy and random

            inverse_slices[:, 0] += 1     # excluding the end of pattern and those action seperators
            slices[:, 0] += 1            # excluding the end of pattern and those action seperators
            
            if inverse_slices.numel() > 0:
                #inverse_slices = torch.cat([torch.arange(*slicee) for slicee in inverse_slices])
                inverse_slices = inverse_slices.int()
            else:
                inverse_slices =None
            #slices[:, 0] +=1
            if slices.numel() > 0:
                slices = torch.cat([torch.arange(*slicee) for slicee in slices])   #creating indices for subsetting the labels, by filling inside of the slices
                slices = slices.int()
                label[slices] = -100
            if bad_mask == 0:
                if label[-2:] == torch.tensor([151645, 198], device=labels.device).all():    #ending actions is not valid..
                    labels[-2:] = -100
            batch_slices.append(inverse_slices)
            random_coeffs.append(random_coeff.to(labels.device))
        return labels, batch_slices, random_coeffs

        # for text, token_ids, random_mask in zip(texts, token_ids_list, random_masks):
        #     tokens_random_mask = []
        #     actions = text.split(self.act_sep_token)
        #     split_token_indices = [i for i, x in enumerate(token_ids) if x == self.act_sep_token_id]
        #     end_indices = split_token_indices + [len(token_ids)]
        #     start_indices = [0] + split_token_indices
        #     action_tokens_lens = [end - start for end, start in zip(end_indices, start_indices)]
        #     if len(action_tokens_lens) != len(random_mask):
        #         raise ValueError("Mismatch between number of actions and random_mask length")
        #     for i, seg_len in enumerate(action_tokens_lens):
        #         tokens_random_mask.extend([random_mask[i]] * seg_len)
        #     model_inputs['weights'].append(tokens_random_mask)
        # return model_inputs
            


    def assign_token_weights2(self, text, model_inputs, offsets):
        sentence_spans = []
        model_inputs['weights'] = []
        for sentence in text:
            cursor = 0
            word_span = []
            for word in sentence.split():
                word_span.append((cursor, cursor + len(word)))
                cursor += len(word) + 1
            sentence_spans.append(word_span)
        
        for i, word_spans in enumerate(sentence_spans):
            offset = offsets[i]
            offset_counter = 0
            weights = []
            for j, word_span in enumerate(word_spans):
                #if offset[offset_counter][1] == word_span[1]
                counter = 0
                while (offset_counter + counter < len(offset)) and offset[offset_counter + counter][1] <= word_span[1]:
                    if text[i][word_span[0]: word_span[1]] in ALF_ACTION_LIST:
                        weights.append(2)
                    else:
                        weights.append(1)
                    counter += 1
                offset_counter += counter
            while len(weights) < len(offset):
                weights.append(1)
            model_inputs['weights'].append(weights)
        return model_inputs
    
    def assign_token_weights(self, text, model_inputs, offsets):
        model_inputs['weights'] = []

        for i, sentence in enumerate(text):
            cursor = 0
            word_spans = []
            for word in sentence.split():
                word_spans.append((cursor, cursor + len(word)))
                cursor += len(word) + 1  # +1 for space

            offset = offsets[i]
            offset_counter = 0
            weights = []

            for word_span in word_spans:
                start, end = word_span
                token_weights = []

                # Check if this word is an action
                is_action = sentence[start:end] in ALF_ACTION_LIST
                # Assign weights to all tokens that belong to this word
                counter = 0
                while (offset_counter + counter < len(offset)) and (offset[offset_counter + counter][1] <= end):
                    if offset[offset_counter + counter] == (0, 0):  # special token
                        token_weights.append(1)
                    else:
                        token_weights.append(2 if is_action else 1)
                    counter += 1

                # Add final token that ends the word
                # if (offset_counter + counter < len(offset)):
                #     token_weights.append(2 if is_action else 1)
                #     counter += 1

                offset_counter += counter
                weights.extend(token_weights)

            # Fill with default weight if any tokens are left (e.g. punctuation or special tokens)
            while len(weights) < len(offset):
                weights.append(1)

            model_inputs['weights'].append(weights)

        return model_inputs