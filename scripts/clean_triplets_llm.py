import os
import sys
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("clean_triplets_llm")

# Add src/ to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.models.llm_server import LLMClient

# Initialize LLM client (using Groq key rotation)
logger.info("Initializing LLMClient with Llama-3.3-70B model...")
llm_client = LLMClient(
    model_name="llama-3.3-70b-versatile",
    backend="openai",
    temperature=0.0,  # Zero temperature for deterministic extraction & classification
    max_tokens=2048
)

SYSTEM_PROMPT = """You are a strict Data Quality Auditor for Information Retrieval (IR) and RAG datasets. Your job is to clean and validate a messy triplet containing (query, positive, negative) across various domains.

You must follow these strict rules for ALL domains:
1. NO CONTEXT EXTENSION (ANTI-HALLUCINATION): You must ONLY use the factual information present in the raw input text. Do NOT add outside knowledge, do NOT predict future events if not mentioned, and do NOT fabricate definitions. If the input text is stupid or weird, just clean the formatting but DO NOT invent facts to make it look better.

2. TEXT CLEANING LAYER: Strip out all web UI junk, navigation links, and formatting noise (e.g., "click here", "try model", "dark mode available", "common questions", "giao diện", "menu buttons"). Retain only semantically rich sentences.

3. IR LOGIC AUDITING (TRIPLET FIXING):
   - 'positive' MUST directly contain the answer to the 'query'.
   - 'negative' MUST be a true Hard Negative. Check if the 'negative' text actually answers the query (creating conflicting labels). If it does, you MUST mutate it into a true hard negative by doing one of the following:
     a) Keep the core entities/domain but select/retain parts that talk about a completely different sub-topic.
     b) If the text is too short to mutate, rewrite the 'query' slightly so that 'positive' remains highly relevant and 'negative' becomes strictly irrelevant.

4. FACTUAL HALLUCINATION AUDITING: If the input text contains severe factual hallucinations, incorrect technical definitions (e.g. defining DeepSeek as a CNN/RNN predictive model for climate/weather forecasting or robotics navigation rather than a Transformer LLM/MoE), or other obvious AI-generated garbage facts, set 'is_hallucinated_or_factual_error' to true. Otherwise, set it to false.

Return ONLY a valid JSON object matching this schema precisely, without any conversational fluff:
{
  "query": "Cleaned, highly focused query",
  "positive": "Cleaned text that strictly and factually answers the query based ONLY on the input",
  "negative": "Cleaned text that is topically related but does NOT answer the query based ONLY on the input",
  "is_hallucinated_or_factual_error": true
}"""

def audit_triplet(triplet: Dict[str, str]) -> Dict[str, Any]:
    """Execute the unified Closed-Domain Auditor check on the triplet."""
    try:
        user_content = json.dumps({
            "query": triplet["query"],
            "positive": triplet["positive"],
            "negative": triplet["negative"]
        }, ensure_ascii=False)
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
        
        resp = llm_client.generate(messages).strip()
        
        # Parse JSON
        start = resp.find("{")
        end = resp.rfind("}")
        if start != -1 and end != -1:
            json_str = resp[start:end+1]
            data = json.loads(json_str)
            return data
        else:
            logger.error(f"Failed to find JSON in response: {resp}")
            return {
                "query": triplet["query"],
                "positive": triplet["positive"],
                "negative": triplet["negative"],
                "is_hallucinated_or_factual_error": False
            }
    except Exception as e:
        logger.error(f"Error auditing triplet: {e}")
        return {
            "query": triplet["query"],
            "positive": triplet["positive"],
            "negative": triplet["negative"],
            "is_hallucinated_or_factual_error": False
        }

def main():
    backup_file = "data/training/reranker/triplets_raw_backup.jsonl"
    output_file = "data/training/reranker/triplets.jsonl"
    
    if not os.path.exists(backup_file):
        logger.error(f"Raw backup file not found at {backup_file}!")
        sys.exit(1)
        
    # Load raw triplets from backup
    triplets: List[Dict[str, str]] = []
    with open(backup_file, "r", encoding="utf-8") as f_in:
        for line in f_in:
            if line.strip():
                triplets.append(json.loads(line))
                
    logger.info(f"Loaded {len(triplets)} raw triplets from {backup_file}.")
    
    # Delete existing output triplets file if it exists to start fresh
    if os.path.exists(output_file):
        logger.info(f"Deleting existing output file {output_file} to start clean purification...")
        os.remove(output_file)
        
    audited_triplets: List[Dict[str, Any]] = []
    completed_audits = 0
    discarded_hallucinations = 0
    
    logger.info("Starting Closed-Domain Triplet Auditing using Llama-3 70B (8 parallel threads)...")
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_triplet = {executor.submit(audit_triplet, t): t for t in triplets}
        for future in as_completed(future_to_triplet):
            try:
                audited = future.result()
                is_hallucinated = audited.get("is_hallucinated_or_factual_error", False)
                
                # Check for other severe hallucination patterns in the cleaned text as a safety fallback
                pos_lower = audited.get("positive", "").lower()
                neg_lower = audited.get("negative", "").lower()
                q_lower = audited.get("query", "").lower()
                
                has_hallucinated_words = (
                    "climate modeling" in pos_lower or "climate modeling" in neg_lower or
                    "predictive model that uses" in pos_lower or "predictive model that uses" in neg_lower or
                    "weather forecasting" in pos_lower or "weather forecasting" in neg_lower
                )
                
                if is_hallucinated or has_hallucinated_words:
                    discarded_hallucinations += 1
                    logger.warning(f"[HALLUCINATION PRUNED] Pruned hallucinated triplet for query: '{audited.get('query')}'")
                else:
                    # Clean the positive/negative values slightly to remove lingering trailing quotes or fluff
                    clean_item = {
                        "query": audited.get("query", "").strip(),
                        "positive": audited.get("positive", "").strip(),
                        "negative": audited.get("negative", "").strip()
                    }
                    if clean_item["query"] and clean_item["positive"] and clean_item["negative"]:
                        audited_triplets.append(clean_item)
            except Exception as e:
                logger.error(f"Thread execution failed: {e}")
                
            completed_audits += 1
            if completed_audits % 20 == 0 or completed_audits == len(triplets):
                logger.info(f"Auditing progress: {completed_audits}/{len(triplets)} triplets completed.")
                
    logger.info(f"Finished Closed-Domain Auditing in {time.time() - start_time:.2f} seconds.")
    
    # Save the purified dataset
    logger.info(f"Saving {len(audited_triplets)} high-quality validated triplets to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f_out:
        for item in audited_triplets:
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    logger.info("\n" + "="*50)
    logger.info("CLOSED-DOMAIN PURIFICATION COMPLETE!")
    logger.info(f"Total raw triplets processed: {len(triplets)}")
    logger.info(f"Hallucinated / Factual error triplets pruned: {discarded_hallucinations}")
    logger.info(f"Triplets saved to triplets.jsonl: {len(audited_triplets)}")
    logger.info("="*50 + "\n")

if __name__ == "__main__":
    main()
