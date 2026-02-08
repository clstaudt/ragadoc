"""
PageIndex RAG backend — fully local tree-based, reasoning-based retrieval via Ollama.

Uses the open-source PageIndex library (vendored via git submodule at
vendor/PageIndex) for tree generation, and the official LLM Tree Search
prompt for retrieval. All LLM calls go through the local Ollama instance
via OpenAI-compatible API.

References:
    - PageIndex open-source repo: https://github.com/VectifyAI/PageIndex
    - LLM Tree Search tutorial: https://docs.pageindex.ai/tutorials/tree-search/llm
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from loguru import logger

import tiktoken
import ollama as ollama_client

# ── Vendor path setup ───────────────────────────────────────────
_VENDOR_DIR = str(Path(__file__).resolve().parent.parent / "vendor" / "PageIndex")
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)

# ── Monkey-patch tiktoken for non-OpenAI models ────────────────
# PageIndex calls tiktoken.encoding_for_model(model) in multiple places
# with the opt.model value. Ollama model names (e.g. "olmo-3.1:latest")
# are unknown to tiktoken. Rather than patching individual call sites
# (which is fragile due to Python's import-time name copying), we patch
# tiktoken.encoding_for_model itself to fall back to cl100k_base.
_original_encoding_for_model = tiktoken.encoding_for_model


def _safe_encoding_for_model(model_name: str):
    """tiktoken.encoding_for_model with fallback for non-OpenAI model names."""
    try:
        return _original_encoding_for_model(model_name)
    except (KeyError, Exception):
        return tiktoken.get_encoding("cl100k_base")


tiktoken.encoding_for_model = _safe_encoding_for_model

# ── PageIndex imports (from vendored submodule) ─────────────────
from pageindex import page_index_main  # type: ignore[import-untyped]
import pageindex.utils as _pi_utils  # type: ignore[import-untyped]
from pageindex.utils import (  # type: ignore[import-untyped]
    ConfigLoader as _PageIndexConfigLoader,
    structure_to_list,
    remove_fields,
)


class PageIndexRAGSystem:
    """
    Fully local RAG backend using PageIndex tree generation + LLM tree search.

    - Document processing: PageIndex tree generation (LLM calls → local Ollama)
    - Retrieval: LLM tree search using official PageIndex tutorial prompt (local Ollama)
    - Answer generation: handled by caller (local Ollama, same as vector RAG)
    """

    def __init__(self, ollama_base_url: str, llm_model: str):
        """
        Initialize the PageIndex RAG system.

        Args:
            ollama_base_url: Ollama server URL (e.g. http://localhost:11434)
            llm_model: Model name for tree generation and tree search
        """
        self.ollama_base_url = ollama_base_url
        self.llm_model = llm_model
        self._current_document_id: Optional[str] = None
        self._tree: Optional[list] = None
        self._node_map: Dict[str, dict] = {}  # node_id -> node

        # Point PageIndex's internal openai calls at local Ollama.
        # OPENAI_BASE_URL: the openai Python client reads this automatically.
        # OPENAI_API_KEY:  required by the openai client (Ollama ignores it).
        # CHATGPT_API_KEY: used by PageIndex's utils.py as the default api_key.
        os.environ["OPENAI_BASE_URL"] = f"{ollama_base_url}/v1"
        os.environ["OPENAI_API_KEY"] = "ollama"
        os.environ["CHATGPT_API_KEY"] = "ollama"

        # PageIndex captures CHATGPT_API_KEY at import time; patch the
        # already-imported module-level variable so it's not None.
        _pi_utils.CHATGPT_API_KEY = "ollama"

    @property
    def backend_type(self) -> str:
        return "pageindex"

    @property
    def current_document_id(self) -> Optional[str]:
        return self._current_document_id

    # ── Document Processing ──────────────────────────────────────

    def process_document(
        self, document_text: str, document_id: str, pdf_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a PageIndex tree from a PDF using local Ollama.

        Args:
            document_text: Extracted text (kept for interface compatibility)
            document_id: Unique identifier for the document
            pdf_path: Path to the PDF file (required for PageIndex)

        Returns:
            Dictionary with processing statistics
        """
        if not pdf_path:
            raise ValueError("PageIndex requires a PDF file path (pdf_path)")

        logger.info(
            f"PageIndex: generating tree for document {document_id} "
            f"using model {self.llm_model}"
        )

        # Use ConfigLoader to merge our overrides with the full set of
        # defaults from config.yaml (toc_check_page_num, max_page_num_each_node, etc.)
        config_loader = _PageIndexConfigLoader()
        opt = config_loader.load({
            "model": self.llm_model,
            "if_add_node_id": "yes",
            "if_add_node_summary": "yes",
            "if_add_node_text": "yes",
            "if_add_doc_description": "no",
        })

        # page_index_main returns {"doc_name": ..., "structure": [...]}
        result = page_index_main(pdf_path, opt)

        self._tree = result.get("structure", result)
        # Ensure tree is a list for consistent handling
        if isinstance(self._tree, dict):
            self._tree = [self._tree]

        self._build_node_map()
        self._current_document_id = document_id

        stats = {
            "document_id": document_id,
            "total_nodes": len(self._node_map),
            "backend": "pageindex",
        }
        logger.info(f"PageIndex: tree generated with {stats['total_nodes']} nodes")
        return stats

    def _build_node_map(self):
        """Flatten tree into a node_id -> node lookup dictionary."""
        self._node_map = {}
        if not self._tree:
            return
        for node in structure_to_list(self._tree):
            nid = node.get("node_id")
            if nid:
                self._node_map[nid] = node

    # ── Retrieval ────────────────────────────────────────────────

    def get_retrieval_info(self, question: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve relevant document sections using LLM tree search.

        Sends the tree structure (without full text) and the question to the
        local Ollama model, which reasons about which nodes are relevant.
        Uses the official PageIndex LLM Tree Search tutorial prompt.

        Args:
            question: User's question

        Returns:
            List of retrieval result dicts compatible with vector RAG output shape
        """
        if not self._tree or not self._node_map:
            raise ValueError("No document tree available. Process a document first.")

        # Prepare tree without text for the search prompt (summaries remain)
        tree_for_search = remove_fields(self._tree, fields=["text"])

        # Official prompt from PageIndex LLM Tree Search tutorial
        # https://docs.pageindex.ai/tutorials/tree-search/llm
        search_prompt = (
            "You are given a query and the tree structure of a document.\n"
            "Each node contains a node id, node title, and a corresponding summary.\n"
            "You need to find all nodes that are likely to contain the answer.\n\n"
            f"Query: {question}\n\n"
            f"Document tree structure:\n{json.dumps(tree_for_search, indent=2)}\n\n"
            "Reply in the following JSON format:\n"
            "{\n"
            '    "thinking": "<your reasoning about which nodes are relevant>",\n'
            '    "node_list": ["node_id_1", "node_id_2"]\n'
            "}\n"
            "Directly return the final JSON structure. Do not output anything else."
        )

        logger.debug(f"PageIndex: running LLM tree search for: {question[:80]}...")

        # Call local Ollama with JSON format enforcement
        client = ollama_client.Client(host=self.ollama_base_url)
        response = client.chat(
            model=self.llm_model,
            messages=[{"role": "user", "content": search_prompt}],
            format="json",
        )
        result_text = response["message"]["content"].strip()

        # Parse the LLM's JSON response
        reasoning = ""
        node_ids: List[str] = []
        try:
            parsed = json.loads(result_text)
            node_ids = parsed.get("node_list", [])
            reasoning = parsed.get("thinking", "")
        except json.JSONDecodeError:
            logger.error(
                f"PageIndex: failed to parse tree search JSON: {result_text[:200]}"
            )

        logger.info(f"PageIndex: tree search selected {len(node_ids)} nodes")

        # Build retrieval results from selected nodes
        chunks_info: List[Dict[str, Any]] = []
        for nid in node_ids:
            node = self._node_map.get(str(nid))
            if node and node.get("text"):
                chunks_info.append(
                    {
                        "chunk_id": nid,
                        "text": node["text"],
                        "score": 1.0,  # reasoning-based, no numeric score
                        "metadata": {
                            "title": node.get("title", ""),
                            "node_id": nid,
                            "start_page": node.get("start_index"),
                            "end_page": node.get("end_index"),
                            "reasoning": reasoning,
                        },
                        "length": len(node["text"]),
                    }
                )
        return chunks_info

    # ── Document Management ──────────────────────────────────────

    def load_document(self, document_id: str) -> bool:
        """Check if the requested document is currently loaded (in-memory only)."""
        return self._current_document_id == document_id and self._tree is not None

    def clear_document(self, document_id: str = None):
        """Clear the current document tree from memory."""
        self._tree = None
        self._node_map = {}
        self._current_document_id = None

    def clear_all_documents(self):
        """Clear all documents (same as clear_document for in-memory backend)."""
        self.clear_document()

    def get_available_documents(self) -> List[str]:
        """Return list of available document IDs."""
        if self._current_document_id:
            return [self._current_document_id]
        return []

    def get_tree(self) -> Optional[list]:
        """Return the cached tree structure for UI display."""
        return self._tree

    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the PageIndex RAG system configuration."""
        return {
            "backend": "pageindex",
            "ollama_base_url": self.ollama_base_url,
            "llm_model": self.llm_model,
            "current_document_id": self._current_document_id,
            "total_nodes": len(self._node_map),
            "has_active_document": self._tree is not None,
        }
