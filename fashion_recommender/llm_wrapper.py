from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
import torch
from .base import BaseComponent

class LLMWrapper(BaseComponent):
    def __init__(self, model_id, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            ),
            trust_remote_code=True
        )
        self.pipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer,
                                 max_new_tokens=1024, temperature=0.7, top_k=50, top_p=0.95,
                                 do_sample=True, repetition_penalty=1.1)
        self.chain = HuggingFacePipeline(pipeline=self.pipeline)
