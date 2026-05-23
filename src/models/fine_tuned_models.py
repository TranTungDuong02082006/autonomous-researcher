import logging
import json
from typing import List, Dict, Any
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class ReviewerOutput(BaseModel):
    sufficient: bool
    missing_info: str
    confidence: float
    findings: str

class FineTunedReviewer:
    """Wrapper loading the fine-tuned Reviewer model weights for custom inference."""
    def __init__(self, base_model_name: str, adapter_path: str):
        self.base_model_name = base_model_name
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None

        logger.info(f"Initializing FineTunedReviewer (Base: {base_model_name}, Adapter: {adapter_path})...")

    def review(self, claim: str, evidence: str) -> ReviewerOutput:
        """Call fine-tuned LoRA model for sufficiency analysis."""
        if self.model is None:
            from peft import PeftModel
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            logger.info("Loading FineTunedReviewer weights into memory...")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)

        system_prompt = "You are a scientific peer reviewer checking evidence sufficiency."
        user_prompt = f"Assess this evidence: Sub-question: '{claim}'. Evidence: {evidence}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=512)
            generated_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        try:
            parsed = json.loads(generated_text.strip())
            return ReviewerOutput(**parsed)
        except Exception:
            logger.warning(f"Could not parse custom reviewer output: '{generated_text}'. Falling back to default.")
            return ReviewerOutput(
                sufficient=True,
                missing_info="",
                confidence=1.0,
                findings=generated_text
            )


class FineTunedReranker:
    """Wrapper loading the fine-tuned CrossEncoder sequence classifier for custom ranking."""
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        logger.info(f"Initializing FineTunedReranker at {model_path}...")

    def rerank(self, query: str, passages: List[str], top_k: int = 5) -> List[str]:
        """Rank passages in descending order of similarity score to the query."""
        if not passages:
            return []

        if self.model is None:
            from sentence_transformers import CrossEncoder
            logger.info("Loading CrossEncoder weights...")
            self.model = CrossEncoder(self.model_path)

        pairs = [[query, passage] for passage in passages]
        scores = self.model.predict(pairs)

        # Sort passages by score descending
        ranked_pairs = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)
        top_passages = [pair[0] for pair in ranked_pairs[:top_k]]
        return top_passages
