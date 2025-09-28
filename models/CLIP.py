import torch
from transformers import CLIPTextModel, AutoTokenizer
from config import ModelConfig
class CLIPTextEncoder(torch.nn.Module):
    def __init__(self, model_name=ModelConfig.model_name):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_model = CLIPTextModel.from_pretrained(model_name)

    @torch.no_grad()
    def encode(self, texts, device, max_length=77, return_mask: bool = False):
        toks = self.tokenizer(
            texts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt"
        )
        toks = {k: v.to(device) for k, v in toks.items()}
        out = self.text_model(**toks)
        if return_mask:
            # key_padding_mask expects True at PAD positions
            key_padding_mask = (toks["attention_mask"] == 0)
            return out.last_hidden_state, key_padding_mask  # (B, L, C), (B, L)
        return out.last_hidden_state  # (B, L, C)

    @torch.no_grad()
    def unconditional(self, batch_size, device, max_length=77):
        return self.encode([""] * batch_size, device=device, max_length=max_length)