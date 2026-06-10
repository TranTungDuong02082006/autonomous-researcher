import json
import logging
import os
import random
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

class ReviewerDatasetBuilder:
    def __init__(self, output_dir: str = "data/training/reviewer"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_synthetic_from_traces(self, trace_files: List[str], n_samples: int = 500):
        """Parse TraceLogger execution logs to generate positive/negative evidence reviews."""
        samples = []
        for file in trace_files:
            if not os.path.exists(file):
                continue
            try:
                with open(file, "r", encoding="utf-8") as f:
                    trace_data = json.load(f)
                
                # Extract reviewer steps
                for step in trace_data.get("steps", []):
                    if step.get("agent_name") == "Reviewer" and "peer-review" in step.get("action", "").lower():
                        # Extract claim context and evaluation
                        inp = step.get("input", "")
                        out = step.get("output", {})
                        
                        # Re-format as standard conversational message pair for SFT SFTTrainer
                        chat_format = {
                            "messages": [
                                {"role": "system", "content": "You are a scientific peer reviewer checking evidence sufficiency."},
                                {"role": "user", "content": f"Assess this evidence: {inp}"},
                                {"role": "assistant", "content": json.dumps(out)}
                            ]
                        }
                        samples.append(chat_format)
            except Exception as e:
                logger.error(f"Failed to process trace file {file}: {e}")

        # Add generic synthetic QA samples to ensure dataset is populated
        if len(samples) < 5:
            logger.warning("No sufficient trace files found. Generating dummy reviewer SFT samples.")
            dummy_samples = [
                {
                    "messages": [
                        {"role": "system", "content": "You are a scientific peer reviewer checking evidence sufficiency."},
                        {"role": "user", "content": "Assess this evidence: Sub-question: 'Context length of Llama 3 8B'. Evidence: [1] Llama 3 has a base context window of 8k tokens."},
                        {"role": "assistant", "content": json.dumps({"sufficient": True, "missing_info": None, "confidence": 0.95, "findings": "Llama 3 8B has an 8k token context window."})}
                    ]
                },
                {
                    "messages": [
                        {"role": "system", "content": "You are a scientific peer reviewer checking evidence sufficiency."},
                        {"role": "user", "content": "Assess this evidence: Sub-question: 'Context length of Claude 3 Opus'. Evidence: [1] Claude 3 family was released in 2024 by Anthropic."},
                        {"role": "assistant", "content": json.dumps({"sufficient": False, "missing_info": "The exact token capacity or context window length of Claude 3 Opus is not mentioned in the source.", "confidence": 0.9, "findings": "No window capacity found."})}
                    ]
                }
            ]
            samples.extend(dummy_samples)

        # Train/Val/Test Split
        random.shuffle(samples)
        split_idx = int(0.8 * len(samples))
        train_samples = samples[:split_idx]
        val_samples = samples[split_idx:]

        self._save_jsonl(train_samples, os.path.join(self.output_dir, "train.jsonl"))
        self._save_jsonl(val_samples, os.path.join(self.output_dir, "val.jsonl"))
        logger.info(f"Saved {len(train_samples)} training samples and {len(val_samples)} validation samples to {self.output_dir}")

    def _save_jsonl(self, data: List[Dict[str, Any]], filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


class RerankerDatasetBuilder:
    def __init__(self, output_dir: str = "data/training/reranker"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_domain(self, url: str) -> str:
        if not url:
            return ""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc.lower()
        except Exception:
            return ""

    def _is_english(self, text: str) -> bool:
        if not text:
            return False
        import re
        # Strict pattern to catch non-English alphabets: CJK, Thai, Greek, Cyrillic, Arabic, Hebrew, Devanagari, and Vietnamese accents
        non_english_pattern = re.compile(
            r'['
            r'\u0370-\u03FF'  # Greek
            r'\u0400-\u04FF'  # Cyrillic
            r'\u0590-\u06FF'  # Hebrew, Arabic, Persian
            r'\u0900-\u0DFF'  # Devanagari and other Indian scripts
            r'\u0E00-\u0E7F'  # Thai
            r'\u2E80-\u2EFF\u3000-\u303F\u3040-\u309F\u30A0-\u30FF\u31F0-\u31FF\u3200-\u32FF\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF\uFF00-\uFFEF\uAC00-\uD7AF'  # CJK
            r'àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ'  # Lowercase Vietnamese accents
            r'ÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ'  # Uppercase Vietnamese accents
            r']'
        )
        if non_english_pattern.search(text):
            return False
        return True


    def _clean_emojis_and_icons(self, text: str) -> str:
        if not text:
            return ""
        import re
        # Strip typical emojis and pictographs
        clean = re.sub(r'[\U00010000-\U0010ffff]', '', text)
        # Strip miscellaneous symbols and dingbats
        clean = re.sub(r'[\u2600-\u27BF\u2300-\u23FF\u2b50\u2934\u2935\u2b06\u2b07]', '', clean)
        # Strip common icons
        clean = re.sub(r'[📢🎯🚀🧠🧩⚡⚙️🎉💡🔍📄🏆✨🔥💬📌]', '', clean)
        # Normalize multiple spaces
        clean = re.sub(r' +', ' ', clean)
        return clean.strip()

    def _has_keyword_overlap(self, query: str, text: str) -> bool:
        import re
        q_words = set(re.findall(r'\b\w{4,}\b', query.lower()))
        stop_words = {
            "what", "when", "where", "which", "who", "whom", "this", "that", "these", "those", 
            "have", "with", "from", "about", "into", "through", "during", "before", "after", 
            "above", "below", "some", "such", "other", "than", "then", "very", "explain", "analyze", 
            "investigate", "compare", "describe", "evaluate", "discuss", "optimal", "potential"
        }
        q_keywords = q_words - stop_words
        if not q_keywords:
            return True
        
        text_lower = text.lower()
        for kw in q_keywords:
            if kw in text_lower:
                return True
        return False

    def _are_queries_similar(self, q1: str, q2: str) -> bool:
        if not q1 or not q2:
            return False
        import re
        words1 = set(re.findall(r'\b\w{3,}\b', q1.lower()))
        words2 = set(re.findall(r'\b\w{3,}\b', q2.lower()))
        stop_words = {
            "what", "when", "where", "which", "who", "whom", "this", "that", "these", "those", 
            "have", "with", "from", "about", "into", "through", "during", "before", "after", 
            "above", "below", "some", "such", "other", "than", "then", "very", "explain", "analyze", 
            "investigate", "compare", "describe", "evaluate", "discuss", "optimal", "potential",
            "development", "model", "models", "architectural", "architecture", "analysis", "system"
        }
        k1 = words1 - stop_words
        k2 = words2 - stop_words
        if not k1 or not k2:
            return False
        intersection = k1.intersection(k2)
        overlap = len(intersection) / min(len(k1), len(k2))
        return overlap > 0.35

    def _find_reviewer_step_for_task(self, steps: List[dict], researcher_idx: int, task_name: str) -> Optional[dict]:
        clean_task_name = task_name.lower().strip()
        # Find subsequent step belonging to Reviewer
        for step in steps[researcher_idx + 1:]:
            if step.get("agent_name") == "Reviewer":
                rev_input = step.get("input", "").lower().strip()
                if " (focus on:" in rev_input:
                    rev_input = rev_input.split(" (focus on:")[0].split(" (focus on:")[0].strip()
                if rev_input == clean_task_name or clean_task_name in rev_input or rev_input in clean_task_name:
                    return step
        # Fallback to the next Reviewer step in sequence
        for step in steps[researcher_idx + 1:]:
            if step.get("agent_name") == "Reviewer":
                return step
        return None

    def generate_triplets(self, trace_files: List[str]):
        """Generate positive and hard negative triplets by mining traces using a strict English loosened margin algorithm."""
        triplets = []
        reputable_domains = [
            "wikipedia.org", "wikipedia.cn", "wikipedia", "github.com", "github.io", "github", 
            "arxiv.org", "arxiv", "deepseek.com", "openai.com", "huggingface.co", "huggingface", 
            "microsoft.com", "google.com", "google", "edu", "org", "research", "w3.org"
        ]
        
        # We store all processed tasks globally across all traces for global cross-session hard negative mining
        global_task_pool = []
        
        for file in trace_files:
            if not os.path.exists(file):
                continue
            try:
                with open(file, "r", encoding="utf-8") as f:
                    trace_data = json.load(f)
                
                steps = trace_data.get("steps", [])
                
                # Map of task query to its document lists in this trace
                trace_task_docs = []
                
                for idx, step in enumerate(steps):
                    agent = step.get("agent_name")
                    action = step.get("action", "")
                    
                    # Relaxed check: match if agent is Researcher, or action contains "research" or "search"
                    is_researcher = (agent == "Researcher") or ("executed research" in action.lower()) or ("research" in action.lower())
                    if not is_researcher:
                        continue
                    
                    query = step.get("input", "")
                    if not query:
                        continue
                        
                    # Clean query (remove focus parameters)
                    if " (focus on:" in query.lower():
                        query = query.split(" (Focus on:")[0].split(" (focus on:")[0].strip()
                    
                    # STRICT FILTER: Query must be in English
                    if not self._is_english(query):
                        continue
                    
                    docs = step.get("output", [])
                    if not isinstance(docs, list) or len(docs) == 0:
                        continue
                    
                    # Find valid chunks
                    valid_docs = [d for d in docs if isinstance(d, dict) and d.get("text")]
                    if not valid_docs:
                        continue
                    
                    # Locate corresponding reviewer evaluation for this task
                    reviewer_step = self._find_reviewer_step_for_task(steps, idx, query)
                    reviewer_confidence = 0.5
                    reviewer_sufficient = True
                    if reviewer_step:
                        rev_output = reviewer_step.get("output", {})
                        if isinstance(rev_output, dict):
                            reviewer_confidence = rev_output.get("confidence", 0.5)
                            reviewer_sufficient = rev_output.get("sufficient", True)
                            
                            # STRICT FILTER: If the Reviewer rejected this step as insufficient with low confidence,
                            # it indicates retrieved evidence is noisy/irrelevant. Discard.
                            if not reviewer_sufficient and reviewer_confidence < 0.5:
                                continue
                    
                    task_item = {
                        "query": query,
                        "docs": valid_docs,
                        "reviewer_confidence": reviewer_confidence,
                        "reviewer_sufficient": reviewer_sufficient
                    }
                    trace_task_docs.append(task_item)
                    global_task_pool.append(task_item)
                
                # Mine triplets within this specific trace file
                for task_idx, item in enumerate(trace_task_docs):
                    query = item["query"]
                    docs = item["docs"]
                    rev_conf = item["reviewer_confidence"]
                    rev_suff = item["reviewer_sufficient"]
                    
                    # Score and sort docs for Positive selection
                    scored_docs = []
                    for doc in docs:
                        base_score = doc.get("score", 0.5)
                        if not isinstance(base_score, (int, float)):
                            base_score = 0.5
                        
                        text = doc["text"]
                        # STRICT FILTER: Positive text must be in English
                        if not self._is_english(text):
                            continue
                            
                        # STRICT FILTER: Positive text must have technical keyword alignment with query
                        if not self._has_keyword_overlap(query, text):
                            continue
                            
                        # Domain reputation check
                        domain = self._get_domain(doc.get("source_url", ""))
                        reputable_boost = 0.0
                        if any(rep in domain for rep in reputable_domains):
                            reputable_boost = 0.15
                            
                        # Reviewer confidence & sufficiency reinforcement
                        reviewer_boost = 0.0
                        if rev_conf > 0.7:
                            reviewer_boost += 0.10
                        if rev_suff:
                            reviewer_boost += 0.10
                            
                        final_score = base_score + reputable_boost + reviewer_boost
                        scored_docs.append({
                            "doc": doc,
                            "final_score": final_score
                        })
                    
                    if not scored_docs:
                        continue
                        
                    # Pick top scored chunk as Positive
                    scored_docs.sort(key=lambda x: x["final_score"], reverse=True)
                    positive_doc = self._clean_emojis_and_icons(scored_docs[0]["doc"]["text"])
                    
                    # 1. Negative 1 (In-session): The lowest scoring English chunk from the same search (geometrically related but weak)
                    english_negatives = [d for d in docs if self._is_english(d["text"]) and self._clean_emojis_and_icons(d["text"]) != positive_doc]
                    if english_negatives:
                        sorted_negs = sorted(english_negatives, key=lambda x: x.get("score", 0.5))
                        negative_in_session = self._clean_emojis_and_icons(sorted_negs[0]["text"])
                        triplets.append({
                            "query": query,
                            "positive": positive_doc,
                            "negative": negative_in_session
                        })
            except Exception as e:
                logger.error(f"Failed to process triplets from trace {file}: {e}")
                
        # 3. Global In-batch Hard Negatives: Ensure every query has a hard negative from a different macro question
        if len(global_task_pool) > 1:
            for idx, item in enumerate(global_task_pool):
                query = item["query"]
                docs = item["docs"]
                
                # Retrieve positive doc
                scored_docs = []
                for doc in docs:
                    if not self._is_english(doc["text"]) or not self._has_keyword_overlap(query, doc["text"]):
                        continue
                    base_score = doc.get("score", 0.5)
                    if not isinstance(base_score, (int, float)):
                        base_score = 0.5
                    domain = self._get_domain(doc.get("source_url", ""))
                    reputable_boost = 0.15 if any(rep in domain for rep in reputable_domains) else 0.0
                    reviewer_boost = 0.20 if item["reviewer_sufficient"] else 0.0
                    
                    final_score = base_score + reputable_boost + reviewer_boost
                    scored_docs.append({"doc": doc, "final_score": final_score})
                
                if not scored_docs:
                    continue
                    
                scored_docs.sort(key=lambda x: x["final_score"], reverse=True)
                positive_doc = self._clean_emojis_and_icons(scored_docs[0]["doc"]["text"])
                
                # Pick a random task from a completely different macro question batch (filtering out similar queries)
                other_global_tasks = [t for t in global_task_pool if not self._are_queries_similar(query, t["query"])]
                if other_global_tasks:
                    random_other = random.choice(other_global_tasks)
                    other_english = [d for d in random_other["docs"] if self._is_english(d["text"])]
                    if other_english:
                        other_english_sorted = sorted(other_english, key=lambda x: x.get("score", 0.5), reverse=True)
                        negative_global_batch = self._clean_emojis_and_icons(other_english_sorted[0]["text"])
                        if negative_global_batch != positive_doc:
                            triplets.append({
                                "query": query,
                                "positive": positive_doc,
                                "negative": negative_global_batch
                            })
                            
        # Fallback to simple samples if still empty or too small
        if not triplets:
            logger.warning("No trace triplets mined. Generating dummy triplets.")
            triplets = [
                {
                    "query": "How large is the context window of Qwen 2.5 14B?",
                    "positive": "Qwen 2.5 14B has a context window length of 128k tokens, allowing it to process very long source documents.",
                    "negative": "Qwen 2.5 14B is a dense language model trained by Alibaba Group on a rich multilingual pre-training dataset."
                }
            ]
            
        filepath = os.path.join(self.output_dir, "triplets.jsonl")
        with open(filepath, "w", encoding="utf-8") as f:
            for item in triplets:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                
        logger.info(f"Mined and generated {len(triplets)} contrastive ranking triplets at {filepath}")
