import torch
import torch.nn.functional as F
from torch import autocast
from transformers import CLIPModel, CLIPProcessor
from contextlib import nullcontext



def prepare_inputs(inputs: dict, device: str):
    for k, v in inputs.items():
        inputs[k] = v.to(device) 
    return inputs


class ClipEncoder:
    def __init__(
        self,
        model_name: str,
        device: str = 'cpu',
        dtype: str = 'bf16',
        l2_norm: bool = True, 
        encode_type: str = None,
    ):
        self.device = torch.device(device)
        
        if dtype == 'fp32':
            self.dtype = torch.float32
        if dtype == 'fp16':
            self.dtype = torch.float16
        if dtype == 'bf16':
            self.dtype = torch.bfloat16 # autocast only support bf16 for CPU

        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        self.l2_norm = l2_norm
        self.encode_type = encode_type

        self.model.eval()
        self.model.to(self.device)

    def encode(self, inputs, **kwargs):
        
        if self.dtype == torch.float32:
            context = nullcontext()
        else:
            context = autocast(device_type=self.device.type, dtype=self.dtype)

        with context:
            if self.encode_type == 'text':
                embeddings = self.get_text_features(inputs['text'])
            elif self.encode_type == 'image':
                embeddings = self.get_image_features(inputs['image'])
            else:
                raise NotImplementedError
            
            if self.l2_norm:
                embeddings = F.normalize(embeddings, dim=-1, p=2.0)
        
        embeddings = embeddings.detach().to(torch.float16).cpu().numpy()

        return embeddings
    
    @torch.no_grad()
    def get_text_features(self, inputs):
        model_inputs = prepare_inputs(inputs, device=self.device)
        embeddings = self.model.get_text_features(**model_inputs)
        return embeddings
    
    @torch.no_grad()
    def get_image_features(self, inputs):
        model_inputs = prepare_inputs(inputs, self.device)
        embeddings = self.model.get_image_features(**model_inputs)
        return embeddings