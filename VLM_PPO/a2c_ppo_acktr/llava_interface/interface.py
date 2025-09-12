import torch
import math
from qwen_vl_utils import process_vision_info
import torchvision.transforms as T
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def llava_generate(value_model, tokenizer, input_ids, image_tensor, args):
    base = value_model.base
    image_tensor = image_tensor.to(base.device, dtype = base.dtype)
    _, _, _, _, inputs_embeds, _ = base.prepare_inputs_labels_for_multimodal(input_ids.to(base.device), None, None, None, None, image_tensor)
    inputs_embeds = inputs_embeds.to(base.device, dtype = base.dtype)
    with torch.inference_mode():
        outputs = base.generate(
        inputs_embeds = inputs_embeds,
        do_sample=True,
        temperature=args.temperature,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        use_cache=True,
        output_scores=True,
        output_hidden_states=True,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.eos_token_id,)
        output_ids = outputs['sequences']
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    padded_output_ids = torch.zeros(output_ids.size(0), 2*args.max_new_tokens).to(dtype=output_ids.dtype, device = output_ids.device)
    padded_output_ids[:, :output_ids.size(1)] = output_ids
    with torch.no_grad():
        values, sum_log_probs, action_tokens_log_prob = llava_evaluate(value_model, input_ids, padded_output_ids, image_tensor, args.temperature, args.thought_prob_coef)
    return values, padded_output_ids, outputs, sum_log_probs, action_tokens_log_prob

def llava_evaluate(value_model, input_ids, output_ids, image_tensor, temperature, thought_prob_coef):
    if output_ids.size(0) != 1:
        input_ids = input_ids.broadcast_to(output_ids.size(0), input_ids.size(-1))
    base = value_model.base
    image_tensor = image_tensor.to(base.device, dtype=base.dtype)
    output_ids = output_ids.to(base.device)
    input_ids = input_ids.to(base.device)
    _, _, _, _, inputs_embeds, _ = base.prepare_inputs_labels_for_multimodal(torch.cat([input_ids, output_ids], dim = 1), None, None, None, None, image_tensor)

    #calling the model
    inputs_embeds = inputs_embeds.to(base.device, dtype = base.dtype)
    #omit the first output token
    outputs = base(
        inputs_embeds = inputs_embeds,
        output_hidden_states = True,
        )
    scores = outputs.logits

    input_token_len = inputs_embeds.shape[1] - output_ids.shape[1]
    hidden_states = outputs.hidden_states[-1][:, input_token_len-1]
    values = value_model.value_head(hidden_states)
    scores = scores * (1/temperature)
    scores = scores.to(torch.float32)
    log_probs = torch.nn.functional.log_softmax(scores, dim=-1)
    log_probs = log_probs.to(torch.bfloat16)
    output_ids_mask = (output_ids != 0)[:, 1:]
    selected_log_probs = output_ids_mask*torch.take_along_dim(log_probs[:, input_token_len:-1], output_ids[:,1:].unsqueeze(2), dim = 2).squeeze(2)
    unfolded = output_ids.unfold(dimension=-1, size=3, step=1)
    target = torch.tensor([29908,2467,1115]).to(base.device)
    # tokens for text string:'"action":' (torch.tensor([[29908,2467,1115]]))
    matches = (unfolded == target).all(dim = -1)
    match_index = matches.nonzero(as_tuple=True)[-1]
    if match_index.shape[0] >= 1:
        match_index = match_index[-1].unsqueeze(0)
    else:
        try:
            match_index = output_ids_mask.nonzero(as_tuple=False)[-4,1]
        except:
            sum_log_prob = torch.tensor([-2]).to(base.device)
            action_tokens_log_prob = torch.tensor([-1]).to(base.device)
            return values, sum_log_prob, action_tokens_log_prob
    ## omitting the second token for calculating log prob, because its logprb is very very small
    thought_log_prob = torch.sum(selected_log_probs[:,1:match_index-1], dim = 1)
    action_tokens_log_prob = torch.sum(selected_log_probs[:,match_index-1:], dim = 1)
    sum_log_prob = thought_prob_coef*thought_log_prob + action_tokens_log_prob
    return values, sum_log_prob, action_tokens_log_prob


def qwen_generate(value_model, processor, text, image, args):
    """
    Inference using QWEN-VL model. Decides on action and gives action value.

    Inputs:
    value_model: model - value model
    processor: hf processor - processor for processing multimodal data
    text: string - textual prompt
    image: tensor - image
    args: arg dict - arguments including generation arguments

    returns:
    values: tensor - action value from value model
    padded_output_ids: tensor - output ids
    outputs: List[string] - textual generated output
    sum_log_probs: tensor - sum of action and thoughts token log probability
    action_tokens_log_prob: tensor - log probability of action tokens only
    """
    input = qwen_process(processor, text, image)
    base = value_model.base
    input = input.to(base.device)
    input_ids = input.input_ids
    with torch.inference_mode():
        outputs = base.generate(
        **input,
        do_sample=True,
        temperature=args.temperature,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
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
    if args.action_sampling:
        return outputs, output_ids, output_ids_trimmed
    else:
        padded_output_ids_trimmed = pad_sequence(output_ids_trimmed, batch_first=True, padding_value=0)
        padded_output_ids = torch.full((output_ids.size(0), 2*args.max_new_tokens), 151643, dtype=output_ids.dtype, device = output_ids.device) #151643 is pad token in qwen2vl #TODO hardcoded
        
        
        padded_output_ids[:, :padded_output_ids_trimmed.size(1)] = padded_output_ids_trimmed
        with torch.no_grad():
            values, sum_log_probs, action_tokens_log_prob = qwen_evaluate(value_model, padded_output_ids, args.temperature, args.thought_prob_coef, processor, input=input  )
        return values, padded_output_ids, outputs, sum_log_probs, action_tokens_log_prob

def qwen_generate_batch(value_model, processor, text, image, args):
    """
    Similar to 'qwen_generate' but for batched inputs.
    #recommended to only be used fro GRPO generation
    """
    
    input = qwen_batch_process(processor, text, image)
    base = value_model.base
    input = input.to(base.device)

    input_ids = input.input_ids
    with torch.inference_mode():
        outs = base.generate(
        **input,
        do_sample=True,
        temperature=args.temperature, #temperature of 1.0 is recommended
        top_p=0.8,  #TODO  hardcoded
        num_beams=args.num_beams,    #num_beams 1 is recommended
        max_new_tokens=args.max_new_tokens,
        use_cache=True,
        output_scores=True,
        output_hidden_states=True,
        return_dict_in_generate=True,
        pad_token_id=processor.tokenizer.eos_token_id,)
        output_ids = outs['sequences'] 
    output_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, output_ids)
                ]
    outputs = processor.batch_decode(
                        output_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )
    
    padded_output_ids_trimmed = pad_sequence(output_ids_trimmed, batch_first=True, padding_value=0)
    padded_output_ids = torch.full((output_ids.size(0), 2*args.max_new_tokens), 151643, dtype=output_ids.dtype, device = output_ids.device) #151643 is pad token in qwen2vl #TODO hardcoded
    
    
    padded_output_ids[:, :padded_output_ids_trimmed.size(1)] = padded_output_ids_trimmed
    with torch.no_grad():
        _, sum_log_probs, action_tokens_log_prob = qwen_evaluate(value_model, padded_output_ids, args.temperature, args.thought_prob_coef, processor, input=input  )
    return outputs, padded_output_ids, sum_log_probs


def qwen_calc_utility(value_model, processor, text, images, args):
    """
    Similar to 'qwen_generate' but for getting the action utility.

    Inputs:
    value_model: model - value model
    processor: hf processor - processor for processing multimodal data
    text: string - textual prompt
    images: List[tensors] - list including before and after images
    args: arg dict - arguments including the generation args
    """
    input = qwen_process_multiImg(processor, text, images)
    base = value_model.base
    input = input.to(base.device)
    input_ids = input.input_ids
    with torch.inference_mode():
        outputs = base.generate(
        **input,
        do_sample=False,
        temperature=args.temperature,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
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


def qwen_calc_utility_batch(value_model, processor, text, images, args):
    """
    Same as 'qwen_calc_utility' but for batch mode.
    """
    input = qwen_batch_process_multiIm(processor, text, images)
    base = value_model.base
    input = input.to(base.device)
    input_ids = input.input_ids
    with torch.inference_mode():
        outputs = base.generate(
        **input,
        do_sample=False,
        temperature=args.temperature,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
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




def qwen_evaluate(value_model, output_ids, temperature, thought_prob_coef, processor, text=None, image=None, input=None, value_only=False):
    """
    Gets action value and action log probability by calculating it based on generated tokens probabilities.

    inputs:
    value_model
    output_ids: tensor - generated output from the prompt
    temperature: float - temperature used for generation of the output_ids
    thought_prob_coef: float - coefficient used for scaling the log probabilities of thought tokens
    processor: hf processor - processor for processing multimodal data
    text: string - textual prompt used for generating the output_ids
    image: tensor - state image
    input: tensor - input ids  used for generating output_ids. processed form of prompt. if it exists the text and image are not required
    value_only: boolean - if True the value won't be calculated and 0 will be returned

    returns:
    values: tensor - action value
    sum_log_prob: tensor - sum of log probs of action and thoughts tokens
    action_tokens_log_prob: tensor - sum of log probs of action tokens only
    """
    if input is None:
        input = qwen_process(processor, text, image)
    
    output_ids = output_ids.to(value_model.base.device)
    input = input.to(value_model.base.device)
    if input['input_ids'].size(0) != 1:
        input['input_ids'] = input['input_ids'].broadcast_to(output_ids.size(0), input['input_ids'].size(-1)) #Mking them with same batch size
    input['input_ids'] = torch.cat([input["input_ids"], output_ids], dim=1)
    input['attention_mask'] = torch.ones_like(input['input_ids'], dtype=torch.long).to(input['input_ids'].device)
    input = input.to(value_model.base.device)


    base = value_model.base
    
    
    outputs = base(
        **input,
        output_hidden_states = True,
        )
    scores = outputs.logits

    input_token_len = input['input_ids'].shape[1] - output_ids.shape[1]
    if value_model.value_head is not None:
        hidden_states = outputs.hidden_states[-1][:, input_token_len-1]
        
        values = value_model.get_value(hidden_states)
    else:
        values = 0
    if value_only:
        return values
    
    scores = scores[:, input_token_len:-1, :]
    scores = scores * (1/temperature)
    scores = scores.to(torch.float32)
    log_probs = torch.nn.functional.log_softmax(scores, dim=-1)
    log_probs = log_probs.to(torch.bfloat16)
    # omit the first outputted id which is decoder start token
    output_ids_mask = (output_ids != 151643)[:, 1:]  #151643 is QWEN model's padding
    ## selected_log_probs counts the log prob of the first token
    
    selected_log_probs = output_ids_mask*torch.take_along_dim(log_probs, output_ids[:,1:].unsqueeze(2), dim = 2).squeeze(2)
    unfolded = output_ids.unfold(dimension=-1, size=2, step=1)
    # the text string '"action":' corresponts to this sequence of tokens: (torch.tensor([[1311, 788]]))  in qwen2vl tokenizer #TODO hardcoded
    target = torch.tensor([1311, 788]).to(base.device)
    matches = (unfolded == target).all(dim = -1)
    match_index = matches.nonzero(as_tuple=True)[-1]
    
    del scores

    if match_index.shape[0] >= 1:
        ## if we find multuple patterns, we will take the last one, and make it size torch.Size([1])
        match_index = match_index[-1].unsqueeze(0)
    else:
        ## if we don't find any pattern, we will take the last 4 tokens, as "action tokens"
        try:
            match_index = output_ids_mask.nonzero(as_tuple=False)[-4,1]
        except:
            sum_log_prob = torch.tensor([-2]).to(base.device)
            action_tokens_log_prob = torch.tensor([-1]).to(base.device)
            return values, sum_log_prob, action_tokens_log_prob
    ## omitting the second token for calculating log prob, because its logprb is very very small
    thought_log_prob = torch.sum(selected_log_probs[:,1:match_index-1], dim = 1)
    
    action_tokens_log_prob = torch.sum(selected_log_probs[:,match_index-1:], dim = 1)
    sum_log_prob = thought_prob_coef*thought_log_prob + action_tokens_log_prob
    return values, sum_log_prob, action_tokens_log_prob


def qwen_evaluate_batch(value_model, output_ids, temperature, thought_prob_coef, processor, text=None, image=None, input=None, grpo=False):
    """
    Same as qwen_evaluate but for batch inference.
    !!!output of the two are not consistent !!!!! #TODO
    """
    if input is None:
        input = qwen_batch_process(processor, text, image)
    
    output_ids = output_ids.to(value_model.base.device)
    input = input.to(value_model.base.device)
    
    if input['input_ids'].size(0) != 1:
        input['input_ids'] = input['input_ids'].repeat(output_ids.size(0), 1)
    
    input['input_ids'] = torch.cat([input["input_ids"], output_ids], dim=1)
    input['attention_mask'] = torch.ones_like(input['input_ids'], dtype=torch.long).to(input['input_ids'].device)
    input = input.to(value_model.base.device)


    base = value_model.base
    
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    
    outputs = base(
        **input,
        output_hidden_states = not grpo,
        )
    scores = outputs.logits

    input_token_len = input['input_ids'].shape[1] - output_ids.shape[1]
    if not grpo:
        hidden_states = outputs.hidden_states[-1][:, input_token_len-1]
        values = value_model.get_value(hidden_states)
    else:
        values = torch.tensor([0])
    
    
    scores = scores * (1/temperature)
    scores = scores.to(torch.float32)
    log_probs = torch.nn.functional.log_softmax(scores, dim=-1)
    log_probs = log_probs.to(torch.bfloat16)
    # omit the first outputted id which is decoder start token
    output_ids_mask = (output_ids != 151643)[:, 1:]
    ## selected_log_probs counts the log prob of the first token
    
    selected_log_probs = output_ids_mask*torch.take_along_dim(log_probs[:, input_token_len:-1], output_ids[:,1:].unsqueeze(2), dim = 2).squeeze(2)
    unfolded = output_ids.unfold(dimension=-1, size=2, step=1)
    # the text string '"action":' corresponts to this sequence of tokens: (torch.tensor([[1311]]))  in qwen2vl tokenizer
    target = torch.tensor([1311, 788]).to(base.device)
    matches = (unfolded == target).all(dim = -1)
    
    match_index = matches.nonzero(as_tuple=True)[-1]
    
    if match_index.shape[0] >= 1:
        ## if we find multuple patterns, we will take the last one, and make it size torch.Size([1])
        match_index = match_index[-1].unsqueeze(0)
    else:
        ## if we don't find any pattern, we will take the last 4 tokens, as "action tokens"
        try:
            match_index = output_ids_mask.nonzero(as_tuple=False)[-4,1]
        except:
            sum_log_prob = torch.tensor([-2]).to(base.device)
            action_tokens_log_prob = torch.tensor([-1]).to(base.device)
            return values, sum_log_prob, action_tokens_log_prob
    ## omitting the second token for calculating log prob, because its logprb is very very small
    thought_log_prob = torch.sum(selected_log_probs[:,1:match_index-1], dim = 1)

    action_tokens_log_prob = torch.sum(selected_log_probs[:,match_index-1:], dim = 1)
    sum_log_prob = thought_prob_coef*thought_log_prob + action_tokens_log_prob
    return values, sum_log_prob, action_tokens_log_prob




def qwen_process(processor, text, image=None):
    """
    processes the input (text and image) and prepare it for QWEN-VL model.

    inputs:
    processor: hf processor - processor for processing multimodal inputs
    text: string - textual prompt
    image: tensor - image, it will be converted to PIL image required by QWEN model

    outputs:
    input: dict - prompt prepared, tokenized, and processed for QWEN-VL model inference
    """
    messages=[]
    message={"role":"user"}
    if image is not None:
        if isinstance(image, torch.Tensor):
            image_tensor = image.squeeze(0).permute(2,0,1).float()
            if image_tensor.max() <= 1.0:
                image_tensor = (image_tensor * 255).byte()
            to_pil = T.ToPILImage()
            image = to_pil(image_tensor)
        content= [
            {"type": "image",
            "image": image,},
            {"type": "text", "text": text},
        ]
    else:
        content= [
            {"type": "text", "text": text},
        ]
    message["content"]=content
    messages.append(message)
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    input = processor(text = prompt_text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            padding_side="left",
            return_tensors="pt",)
    return input

def qwen_process_multiImg(processor, text, images):
    """
    Similar to 'qwen_process' but supporting multi images

    Inputs:
    similar to qwen_process except:
    images: List[tensors]
    """
    messages=[]
    message={"role":"user"}
    content= [
            {"type": "text", "text": text},
        ]
    for i, image in enumerate(images):
        if isinstance(image, torch.Tensor):
            image_tensor = image.squeeze(0).permute(2,0,1).float()
            if image_tensor.max() <= 1.0:
                image_tensor = (image_tensor * 255).byte()
            to_pil = T.ToPILImage()
            image = to_pil(image_tensor)
        content.append(
            {"type": "image",
            "image": image,}
        )
    message["content"]=content
    messages.append(message)
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    input = processor(text = prompt_text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            padding_side="left",
            return_tensors="pt",)
    return input



def qwen_batch_process(processor, texts, images=None):
    """
    Batch processing for Qwen multimodal model.
    Args:
        processor: Hugging Face Qwen processor
        texts (List[str]): List of texts
        images (List[Union[PIL.Image, Tensor, None]]): List of images or None

    Returns:
        Batch dict with tokenized inputs
    """
    assert isinstance(texts, list), "texts should be a list of strings"
    if images is not None:
        assert len(texts) == len(images), "texts and images must be the same length"

    messages_batch = []
    image_inputs_batch = []
    video_inputs_batch = []

    to_pil = T.ToPILImage()

    for i, text in enumerate(texts):
        image = images[i] if images is not None else None

        #processor requires image in PIL format
        if isinstance(image, torch.Tensor):
            image_tensor = image.squeeze(0).permute(2, 0, 1).float()
            if image_tensor.max() <= 1.0:
                image_tensor = (image_tensor * 255).byte()
            image = to_pil(image_tensor)
        
        message = {"role": "user"}
        if image is not None:
            content = [
                {"type": "image", "image": image},
                {"type": "text", "text": text},
            ]
        else:
            content = [{"type": "text", "text": text}]

        message["content"] = content
        messages_batch.append([message])  #Note: outer list because processor expects batch of conversations

    #prepare multimodal for input into processor
    prompt_texts = [
        processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in messages_batch
    ]
    image_inputs_batch, _ = zip(*[
        process_vision_info(messages) for messages in messages_batch
    ])

    inputs = processor(
        text=prompt_texts,
        images=list(image_inputs_batch),
        #videos=list(video_inputs_batch),
        padding=True,
        return_tensors="pt",
    )
    return inputs

def qwen_batch_process_multiIm(processor, texts, images=None):
    """
    Batch processing for Qwen multimodal model.
    Args:
        processor: Hugging Face Qwen processor
        texts (List[str]): List of texts
        images (List[Union[PIL.Image, Tensor, None]]): List of images or None

    Returns:
        Batch dict with tokenized inputs
    """
    assert isinstance(texts, list), "texts should be a list of strings"
    if images is not None:
        assert len(texts) == len(images), "texts and images must be the same length"

    messages_batch = []
    image_inputs_batch = []
    video_inputs_batch = []

    to_pil = T.ToPILImage()

    for i, text in enumerate(texts):
        images_ = images[i] if images is not None else None

        #processor requires image in PIL format
        content = []
        message = {"role": "user"}
        for image in images_:
            if isinstance(image, torch.Tensor):
                image_tensor = image.squeeze(0).permute(2, 0, 1).float()
                if image_tensor.max() <= 1.0:
                    image_tensor = (image_tensor * 255).byte()
                image = to_pil(image_tensor)
            if image is not None:
                content.append(
                    {"type": "image", "image": image}
                            )
        content.append({"type": "text", "text": text})
        message["content"] = content
        messages_batch.append([message])  #Note: outer list because processor expects batch of conversations

    #prepare multimodal for input into processor
    prompt_texts = [
        processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in messages_batch
    ]
    image_inputs_batch, _ = zip(*[
        process_vision_info(messages) for messages in messages_batch
    ])

    inputs = processor(
        text=prompt_texts,
        images=list(image_inputs_batch),
        #videos=list(video_inputs_batch),
        padding=True,
        return_tensors="pt",
    )
    return inputs


def format_data_sft(sample):
    """
    formats data for supervised fine-tuning of QWEN-vl models
    
    Inputs:
    sample: dict - should include followings:
    sample["image"]:  should be a list of tensor or pilImage
    sample["query"]: string - the question
    sample["label"]: string - ground truth label
    """
    input_images = sample["image"]
    images = []
    for image in input_images:
        if isinstance(image, torch.Tensor):
                if image.ndim == 4:
                    image_tensor = image.squeeze(0).permute(2,0,1).float()
                else:
                    image_tensor = image.permute(2, 0, 1).float()
                
                if image_tensor.max().item() <= 1.0:
                    image_tensor = (image_tensor * 255)
                
                to_pil = T.ToPILImage()
                image = to_pil(image_tensor)
                images.append(image)
        elif isinstance(image, Image.Image):
            images.append(image)
    contents = []
    for img in images:
        contents.append({
            "type": "image",
            "image": img,
        })
    contents.append(
        {
            "type": "text",
            "text": sample["query"],
        }
    )
    return [
        {
            "role": "user",
            "content": contents,
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": sample["label"]}],
        },
    ]
