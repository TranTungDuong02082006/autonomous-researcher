import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class KeyRotator:
    def __init__(self):
        primary_key = os.getenv("GROQ_API_KEY")
        keys_str = os.getenv("GROQ_API_KEYS", "")
        
        self.keys = []
        if primary_key:
            self.keys.append(primary_key)
            
        if keys_str:
            for k in keys_str.split(","):
                k = k.strip().replace('"', '').replace("'", "")
                if k and k not in self.keys:
                    self.keys.append(k)
                    
        self.index = 0
        logger.info(f"KeyRotator initialized with {len(self.keys)} Groq API keys.")

    def get_key(self) -> Optional[str]:
        if not self.keys:
            return None
        return self.keys[self.index]

    def rotate(self) -> Optional[str]:
        if not self.keys:
            return None
        self.index = (self.index + 1) % len(self.keys)
        logger.warning(f"[ROTATION] Rate limit hit! Rotating to Groq API Key at index {self.index}: ...{self.keys[self.index][-8:]}")
        return self.keys[self.index]

# Global key rotator instance
_key_rotator = None

def get_key_rotator():
    global _key_rotator
    if _key_rotator is None:
        _key_rotator = KeyRotator()
    return _key_rotator

class LLMClient:
    """
    Unified client wrapper for interacting with LLM models.
    Supports vLLM (via OpenAI compatibility layer), local HuggingFace Transformers,
    or direct API providers like OpenAI/OpenRouter/Gemini/Groq.
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

        # Prioritize environment variables over dummy/default values
        key_from_env = os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")
        
        # Auto-detect Gemini models
        if self.model_name.startswith("gemini-") or self.model_name in ["gemini-1.5-flash", "gemini-1.5-pro"]:
            gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if gemini_key:
                logger.info(f"Gemini model detected. Routing to Google AI Studio for {self.model_name}.")
                self.api_key = gemini_key
                self.api_base = "https://generativelanguage.googleapis.com/v1beta/openai/"
                self.backend = "openai"
            else:
                self.api_key = "dummy-key"
                self.api_base = api_base or "http://localhost:8000/v1"
        else:
            if not api_key or api_key == "dummy-key":
                rotator = get_key_rotator()
                env_key = rotator.get_key()
                self.api_key = env_key or key_from_env or "dummy-key"
            else:
                self.api_key = api_key

            # Auto-detect Groq keys to use Groq's official API base
            default_base = "https://api.groq.com/openai/v1" if (self.api_key and self.api_key.startswith("gsk_")) else "http://localhost:8000/v1"
            base_from_env = os.getenv("OPENAI_API_BASE") or os.getenv("GROQ_API_BASE")
            if not api_base or api_base == "http://localhost:8000/v1":
                self.api_base = base_from_env or default_base
            else:
                self.api_base = api_base

            # Auto-map to a supported Groq model if using Groq and the model is not hosted there
            if self.api_key and self.api_key.startswith("gsk_"):
                groq_models = ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma2-9b-it", "llama-3.1-8b-instant"]
                if self.model_name not in groq_models and not any(m in self.model_name.lower() for m in ["llama", "mixtral", "gemma"]):
                    logger.info(f"Auto-mapping non-Groq model '{self.model_name}' to 'llama-3.3-70b-versatile' for Groq compatibility.")
                    self.model_name = "llama-3.3-70b-versatile"

        self.client = None
        self.pipeline = None

        if self.backend in ["vllm", "openai"]:
            try:
                from openai import OpenAI
                # Set max_retries=0 to immediately raise RateLimitError, letting our KeyRotator rotate keys instantly!
                self.client = OpenAI(base_url=self.api_base, api_key=self.api_key, max_retries=0)
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

    def _execute_with_retry(self, func, *args, **kwargs):
        """Wrapper to execute an API call with automatic Groq key rotation on 429 rate limit errors."""
        max_attempts = 15
        for attempt in range(max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = "429" in err_str or "rate limit" in err_str or "too many requests" in err_str
                
                # Check if this is a Groq key to rotate
                rotator = get_key_rotator()
                if is_rate_limit and rotator and len(rotator.keys) > 1 and not self.model_name.startswith("gemini-"):
                    new_key = rotator.rotate()
                    self.api_key = new_key
                    # Recreate client with rotated key
                    if self.backend in ["vllm", "openai"]:
                        try:
                            from openai import OpenAI
                            self.client = OpenAI(base_url=self.api_base, api_key=self.api_key, max_retries=0)
                        except Exception:
                            pass
                    logger.info(f"[RETRY] Retrying API call with rotated key (attempt {attempt + 1}/{max_attempts})...")
                    continue
                else:
                    raise e

    def generate(
        self, 
        messages: List[Dict[str, str]], 
        temperature: Optional[float] = None, 
        max_tokens: Optional[int] = None
    ) -> str:
        """Standard generation method taking a list of message dicts (role, content)."""
        return self._execute_with_retry(self._generate_raw, messages, temperature, max_tokens)

    def _generate_raw(
        self, 
        messages: List[Dict[str, str]], 
        temperature: Optional[float] = None, 
        max_tokens: Optional[int] = None
    ) -> str:
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
        return self._execute_with_retry(self._generate_structured_raw, messages, schema, temperature, max_tokens)

    def _generate_structured_raw(
        self, 
        messages: List[Dict[str, str]], 
        schema: Any, 
        temperature: Optional[float] = None, 
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
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
        raw_output = self._generate_raw(formatted_messages, temperature=temp, max_tokens=tokens)
        return self._parse_json(raw_output)

    def generate_with_tools(
        self, 
        messages: List[Dict[str, str]], 
        tools: List[Dict[str, Any]], 
        temperature: Optional[float] = None, 
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate content with OpenAI-compatible tool/function call formats."""
        return self._execute_with_retry(self._generate_with_tools_raw, messages, tools, temperature, max_tokens)

    def _generate_with_tools_raw(
        self, 
        messages: List[Dict[str, str]], 
        tools: List[Dict[str, Any]], 
        temperature: Optional[float] = None, 
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
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
        raw_output = self._generate_raw(formatted_messages, temperature=temp, max_tokens=tokens)
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
