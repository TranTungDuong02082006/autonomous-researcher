import json
import logging
import os
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class LLMClient:
    """
    Unified client wrapper for interacting with LLM models.
    Supports vLLM (via OpenAI compatibility layer), local HuggingFace Transformers,
    or direct API providers like OpenAI/OpenRouter.
    """
    def __init__(
        self, 
        model_name: str, 
        backend: str = "vllm", 
        quantization: Optional[str] = "4bit",
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ):
        self.model_name = model_name
        self.backend = backend.lower()
        self.quantization = quantization
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Load API details from env vars if not provided
        self.api_base = api_base or os.getenv("OPENAI_API_BASE") or "http://localhost:8000/v1"
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or "dummy-key"

        self.client = None
        self.pipeline = None

        if self.backend in ["vllm", "openai"]:
            try:
                from openai import OpenAI
                self.client = OpenAI(base_url=self.api_base, api_key=self.api_key)
            except ImportError:
                logger.warning("openai package not found, falling back to httpx for API calls.")
                import httpx
                self.client = httpx.Client(headers={"Authorization": f"Bearer {self.api_key}"})
        elif self.backend == "transformers":
            logger.info("Initializing local HuggingFace Transformers pipeline...")
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

            # Config quantization
            bnb_config = None
            if self.quantization == "4bit":
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            elif self.quantization == "8bit":
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
            )
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def generate(
        self, 
        messages: List[Dict[str, str]], 
        temperature: Optional[float] = None, 
        max_tokens: Optional[int] = None
    ) -> str:
        """Standard generation method taking a list of message dicts (role, content)."""
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        if self.backend in ["vllm", "openai"]:
            if hasattr(self.client, "chat"):
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temp,
                    max_tokens=tokens
                )
                return response.choices[0].message.content
            else:
                # httpx fallback
                payload = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": temp,
                    "max_tokens": tokens
                }
                res = self.client.post(f"{self.api_base}/chat/completions", json=payload)
                res.raise_for_status()
                return res.json()["choices"][0]["message"]["content"]

        elif self.backend == "transformers":
            # Convert chat template
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            outputs = self.pipeline(
                prompt,
                max_new_tokens=tokens,
                do_sample=temp > 0,
                temperature=temp if temp > 0 else 1.0,
                pad_token_id=self.tokenizer.eos_token_id
            )
            generated_text = outputs[0]["generated_text"]
            # Extract generation part
            return generated_text[len(prompt):].strip()

    def generate_structured(
        self, 
        messages: List[Dict[str, str]], 
        schema: Any, 
        temperature: Optional[float] = None, 
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """JSON output generation constrained by a Pydantic schema or JSON schema dict."""
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        # Build schema instructions
        schema_dict = schema.model_json_schema() if hasattr(schema, "model_json_schema") else schema
        schema_instruction = (
            f"\n\nYou MUST return a JSON object that adheres exactly to this JSON schema:\n"
            f"{json.dumps(schema_dict, indent=2)}\n"
            f"Do not include any explanation or markdown formatting outside of the JSON block. "
            f"Start your response with '{{' and end with '}}'."
        )
        
        # Clone messages and append instructions to the last user message or system message
        formatted_messages = list(messages)
        if formatted_messages:
            formatted_messages[-1] = {
                "role": formatted_messages[-1]["role"],
                "content": formatted_messages[-1]["content"] + schema_instruction
            }

        if self.backend in ["vllm", "openai"] and hasattr(self.client, "chat"):
            try:
                # Attempt to use JSON mode if supported
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=formatted_messages,
                    temperature=temp,
                    max_tokens=tokens,
                    response_format={"type": "json_object"}
                )
                output_str = response.choices[0].message.content
                return self._parse_json(output_str)
            except Exception as e:
                logger.warning(f"Failed to use OpenAI JSON mode, trying raw extraction: {e}")

        # Fallback raw call and regex parsing
        raw_output = self.generate(formatted_messages, temperature=temp, max_tokens=tokens)
        return self._parse_json(raw_output)

    def generate_with_tools(
        self, 
        messages: List[Dict[str, str]], 
        tools: List[Dict[str, Any]], 
        temperature: Optional[float] = None, 
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate content with OpenAI-compatible tool/function call formats."""
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        if self.backend in ["vllm", "openai"] and hasattr(self.client, "chat"):
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=temp,
                max_tokens=tokens
            )
            message = response.choices[0].message
            if message.tool_calls:
                tool_call = message.tool_calls[0]
                return {
                    "name": tool_call.function.name,
                    "arguments": json.loads(tool_call.function.arguments)
                }
            return {"content": message.content}
        
        # Simple prompting fallback for tools
        tools_instruction = (
            f"\n\nYou are allowed to make a function call. Choose the most relevant function from:\n"
            f"{json.dumps(tools, indent=2)}\n"
            f"If you choose to call a function, respond with JSON format:\n"
            f"{{\"name\": \"function_name\", \"arguments\": {{...}}}}\n"
            f"Otherwise respond with format: {{\"content\": \"your text response\"}}"
        )
        formatted_messages = list(messages)
        formatted_messages[-1] = {
            "role": formatted_messages[-1]["role"],
            "content": formatted_messages[-1]["content"] + tools_instruction
        }
        raw_output = self.generate(formatted_messages, temperature=temp, max_tokens=tokens)
        return self._parse_json(raw_output)

    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Extract and parse JSON from string even if wrapped in markdown codeblocks."""
        clean_text = text.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        elif clean_text.startswith("```"):
            clean_text = clean_text[3:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
        clean_text = clean_text.strip()
        
        # Find first '{' and last '}'
        start_idx = clean_text.find("{")
        end_idx = clean_text.rfind("}")
        if start_idx != -1 and end_idx != -1:
            clean_text = clean_text[start_idx:end_idx + 1]

        try:
            return json.loads(clean_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from text: {text}. Error: {e}")
            raise e
