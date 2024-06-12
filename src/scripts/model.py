import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

def get_model(model_name):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
    )
    model.config.use_cache = False

    return model