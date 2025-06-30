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
ALF_ACTION_LIST=["pass", "goto", "pick", "put", "open", "close", "toggle", "heat", "clean", "cool", "slice", "inventory", "examine", "look"]





class TrajectoryDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings["input_ids"])


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


class Temp_predictor():
    def __init__(self, device, model, processor, epochs=4, temp_model_lr=3e-4, buff_size=160):

        """
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        model = T5Model.from_pretrained("t5-small")
        """
        self.act_sep_token = '<ACT_SEP>'  #special action seperator token
        self.tokenizer = model#T5TokenizerFast.from_pretrained("t5-small", extra_special_tokens={"act_token": self.act_sep_token})
        self.model = model
        #self.model = T5ForConditionalGeneration.from_pretrained("t5-small")
        self.temp_model_optimizer = torch.optim.Adam(self.model.parameters(), lr = temp_model_lr)
        self.device = device
        #self.accelerator = accelerator
        self.epochs = epochs
        
        self.tokenizer.add_special_tokens({'act_token': self.act_sep_token})  #action seperator token
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.act_sep_token_id = self.tokenizer.convert_tokens_to_ids(self.act_sep_token)
        #self.temp_model_optimizer = self.accelerator.prepare(self.temp_model_optimizer)
        self.model.to(self.device)

        #trajectories buffer
        self.input_seq_buffer = deque(maxlen=buff_size)
        self.target_seq_buffer = deque(maxlen=buff_size)


    def preprocess_trajectories(self, buffer_goals, buffer_trajLen, buffer_actions, terminals):
        input_seq = []
        target_seq = []
        counter =0
        for i, len in enumerate(buffer_trajLen):
            acts = buffer_actions[counter: len]
            counter += len
            if terminals[i]:
                self.input_seq_buffer.append(buffer_goals[i])
                flattened = f" {self.act_sep_token} ".join(acts)
                self.target_seq_buffer.append(flattened)
        # input_seq= None
        # target_seq= None
        return list(self.input_seq_buffer), list(self.target_seq_buffer)
    
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
    
    def compute_novelty(self, input, target):
        """
        computes novetly for one trajectory
        input: a string, mainly the trajectory's goal
        target: list of strings, mainly the trajectory's actions
        """
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
        #breakpoint()
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
        return final_loss



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
                        #breakpoint()
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
                #breakpoint()
                # Assign weights to all tokens that belong to this word
                counter = 0
                #breakpoint()
                while (offset_counter + counter < len(offset)) and (offset[offset_counter + counter][1] <= end):
                    #breakpoint()
                    if offset[offset_counter + counter] == (0, 0):  # special token
                        token_weights.append(1)
                    else:
                        token_weights.append(2 if is_action else 1)
                    #breakpoint()
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