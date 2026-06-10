import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.graph.state import Citation
import re

def test_citation_formatting():
    # Verify Citation Pydantic state model works
    c1 = Citation(index=1, url="https://deepseek.com", title="DeepSeek MoE")
    c2 = Citation(index=2, url="https://llama.meta.com", title="Llama 3 Attention")

    assert c1.index == 1
    assert c1.url == "https://deepseek.com"
    assert c1.title == "DeepSeek MoE"

    # Verify formatting references list
    citations = [c1, c2]
    refs_str = "\n\n## References\n"
    for cit in citations:
        refs_str += f"[{cit.index}] [{cit.title}]({cit.url})\n"

    assert "## References" in refs_str
    assert "[1] [DeepSeek MoE](https://deepseek.com)" in refs_str
    assert "[2] [Llama 3 Attention](https://llama.meta.com)" in refs_str

def test_inline_citation_parsing():
    # Simulate a written report body
    report_body = (
        "# DeepSeek-V3 and Llama 3 Comparison\n\n"
        "DeepSeek-V3 implements Multi-head Latent Attention (MLA) to compress KV cache [1]. "
        "Llama 3, on the other hand, utilizes Grouped-Query Attention (GQA) [2]."
    )

    # Search for inline bracket citations
    inline_citations = re.findall(r"\[\d+\]", report_body)
    assert len(inline_citations) == 2
    assert "[1]" in inline_citations
    assert "[2]" in inline_citations
