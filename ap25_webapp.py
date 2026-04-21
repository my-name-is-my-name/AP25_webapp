from __future__ import annotations

import argparse
import gc
import json
import math
import mimetypes
import os
import platform
import sys
import threading
import traceback
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    import pysqlite3  # type: ignore

    sys.modules["sqlite3"] = pysqlite3
except Exception:
    pass

import torch
from huggingface_hub import snapshot_download
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer

from ap25_parser import parse_ap25_for_notebook, to_langchain_documents
from retrieval_engine import AP25RetrievalEngine, chunk_header


WORKDIR = Path(__file__).resolve().parent
MODEL_DIR = WORKDIR / "models"
VECTORSTORE_DIR = WORKDIR / "chroma_db_ap25_bge_m3_searchtext"
COLLECTION_NAME = "ap25_bge_m3_searchtext_v1"
STATIC_DIR = WORKDIR / "webapp_static"
PDF_PATH = WORKDIR / "AP25.pdf"

EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
EMBEDDING_MODEL_DIR = MODEL_DIR / "embedding_bge_m3"
RERANKER_MODEL_DIR = MODEL_DIR / "reranker_bge_v2_m3"

TOP_K_DENSE = 40
TOP_K_LEXICAL = 60
TOP_K_FUSED = 60
TOP_K_FINAL = 8
MAX_PER_PARAGRAPH = 2
MAX_CONTEXT_CHARS = 12000

SYSTEM_PROMPT = (
    "Ты анализируешь нормативный документ АП-25.\n"
    "Используй ТОЛЬКО предоставленные retrieval-чанки как единственный источник информации.\n"
    "Не используй внешние знания, инженерные догадки или типовые требования, если их нет в контексте.\n"
    "Если данных недостаточно, прямо напиши: \"Недостаточно данных.\"\n"
    "Каждый тезис ответа по возможности сопровождай ссылкой на параграф вида §25.xxx.\n"
    "Если в контексте уже есть прямой ответ, отвечай кратко и по существу.\n"
)

GENERATION_CONFIG = {
    "max_new_tokens": 900,
    "do_sample": False,
    "pad_token_id": None,
}

ISSUE_SPOTTING_VALIDATOR_PROMPT = (
    "Ты валидируешь применимость конкретного пункта АП-25 к инженерному кейсу.\n"
    "Используй только текст кейса и текст данного пункта.\n"
    "Не придумывай дополнительные требования и не опирайся на внешние знания.\n"
    "Верни только JSON без markdown и без пояснений вне JSON.\n"
    "Формат JSON: {\"decision\": \"core|supporting|uncertain|not_applicable\", "
    "\"applicability_score\": 0-100, \"confidence\": \"high|medium|low\", \"reason\": \"...\"}.\n"
    "decision=core только если пункт явно и непосредственно относится к кейсу.\n"
    "decision=supporting если пункт полезен для анализа, но не является главным.\n"
    "decision=uncertain если по тексту пункта нельзя уверенно решить вопрос применимости.\n"
    "decision=not_applicable если пункт не относится к кейсу по смыслу."
)

ISSUE_DECISION_ORDER = {
    "core": 0,
    "supporting": 1,
    "uncertain": 2,
    "not_applicable": 3,
}

ISSUE_VALIDATOR_CONFIG = {
    "max_new_tokens": 260,
    "do_sample": False,
    "pad_token_id": None,
}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    repo_id: str | None
    local_dir_name: str
    recommended_for: str
    notes: str
    aliases: tuple[str, ...] = ()
    allow_download: bool = True
    env_var: str | None = None

    @property
    def local_dir(self) -> Path:
        return MODEL_DIR / self.local_dir_name

    def resolved_local_dir(self) -> Path:
        if self.env_var:
            override = os.environ.get(self.env_var)
            if override:
                return Path(override).expanduser().resolve()
        return self.local_dir


MODEL_SPECS = {
    "qwen25_15b_instruct": ModelSpec(
        key="qwen25_15b_instruct",
        display_name="Qwen2.5-1.5B-Instruct",
        repo_id="Qwen/Qwen2.5-1.5B-Instruct",
        local_dir_name="llm_qwen25_15b_instruct",
        recommended_for="быстрые локальные тесты",
        notes="Текущая базовая модель проекта.",
    ),
    "qwen25_7b_instruct": ModelSpec(
        key="qwen25_7b_instruct",
        display_name="Qwen2.5-7B-Instruct",
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        local_dir_name="llm_qwen25_7b_instruct",
        recommended_for="более сильные ответы при умеренной нагрузке",
        notes="Использована как безопасная интерпретация запроса про qwen-7.5b.",
        aliases=("qwen-7.5b", "qwen7.5b", "qwen_7_5b"),
    ),
    "qwen36_35b_a3b": ModelSpec(
        key="qwen36_35b_a3b",
        display_name="Qwen3.6-35B-A3B",
        repo_id=None,
        local_dir_name="llm_qwen36_35b_a3b",
        recommended_for="тяжелые сравнительные тесты на мощной GPU",
        notes="Большая модель. Автоскачивание отключено: ожидается уже готовый локальный путь.",
        aliases=("qwen3.6-35b-a3b", "qwen36-35b-a3b"),
        allow_download=False,
        env_var="AP25_QWEN36_35B_A3B_PATH",
    ),
}

MODEL_ALIASES = {
    alias: spec.key
    for spec in MODEL_SPECS.values()
    for alias in spec.aliases
}


def resolve_model_key(model_key: str | None) -> str:
    if not model_key:
        return "qwen25_15b_instruct"
    if model_key in MODEL_SPECS:
        return model_key
    return MODEL_ALIASES.get(model_key, model_key)


def ensure_model_downloaded(repo_id: str, local_dir: Path) -> Path:
    local_dir = Path(local_dir)
    if local_dir.exists() and any(local_dir.iterdir()):
        return local_dir
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )
    return local_dir


def safe_score(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        numeric = float(value)
    except Exception:
        return default
    if math.isnan(numeric) or math.isinf(numeric):
        return default
    return numeric


def extract_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("JSON object not found")
    return json.loads(text[start : end + 1])


def coerce_issue_decision(value: str) -> str:
    value = (value or "").strip().lower()
    if value in ISSUE_DECISION_ORDER:
        return value
    aliases = {
        "applicable": "supporting",
        "yes": "supporting",
        "no": "not_applicable",
        "n/a": "not_applicable",
    }
    return aliases.get(value, "uncertain")


class ModelManager:
    def __init__(self, device: str):
        self.device = device
        self.current_model_key: str | None = None
        self.current_spec: ModelSpec | None = None
        self.tokenizer = None
        self.model = None
        self.lock = threading.RLock()

    def list_models(self) -> list[dict[str, Any]]:
        items = []
        for spec in MODEL_SPECS.values():
            resolved_dir = spec.resolved_local_dir()
            items.append(
                {
                    "key": spec.key,
                    "display_name": spec.display_name,
                    "repo_id": spec.repo_id,
                    "local_dir": str(resolved_dir),
                    "local_exists": resolved_dir.exists() and any(resolved_dir.iterdir()),
                    "recommended_for": spec.recommended_for,
                    "notes": spec.notes,
                    "loaded": spec.key == self.current_model_key,
                    "allow_download": spec.allow_download,
                    "env_var": spec.env_var,
                }
            )
        return items

    def _target_dtype(self) -> torch.dtype:
        if self.device == "cuda":
            return torch.bfloat16
        return torch.float32

    def unload(self) -> None:
        with self.lock:
            self.model = None
            self.tokenizer = None
            self.current_model_key = None
            self.current_spec = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def load(self, model_key: str, download_if_missing: bool = False) -> dict[str, Any]:
        resolved_key = resolve_model_key(model_key)
        if resolved_key not in MODEL_SPECS:
            raise ValueError(f"Unknown model key: {model_key}")

        spec = MODEL_SPECS[resolved_key]
        with self.lock:
            if self.current_model_key == spec.key and self.model is not None and self.tokenizer is not None:
                return {
                    "status": "already_loaded",
                    "model_key": spec.key,
                    "display_name": spec.display_name,
                }

            model_path = spec.resolved_local_dir()
            if not (model_path.exists() and any(model_path.iterdir())):
                if not download_if_missing:
                    raise FileNotFoundError(
                        f"Model files not found for {spec.display_name}. "
                        "Use download_if_missing=true or load the model on disk first."
                    )
                if not spec.allow_download or not spec.repo_id:
                    raise FileNotFoundError(
                        f"{spec.display_name} is configured for local use only. "
                        f"Point {spec.env_var or 'the model path'} to an existing folder with model files."
                    )
                ensure_model_downloaded(spec.repo_id, model_path)

            self.unload()
            tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)

            if self.device == "cuda":
                model = AutoModelForCausalLM.from_pretrained(
                    str(model_path),
                    torch_dtype=self._target_dtype(),
                    device_map="auto",
                    trust_remote_code=True,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    str(model_path),
                    torch_dtype=self._target_dtype(),
                    trust_remote_code=True,
                )
                model = model.to(self.device)

            model.eval()
            self.tokenizer = tokenizer
            self.model = model
            self.current_model_key = spec.key
            self.current_spec = spec
            return {
                "status": "loaded",
                "model_key": spec.key,
                "display_name": spec.display_name,
            }

    def ensure_default(self) -> None:
        if self.model is None or self.tokenizer is None:
            self.load("qwen25_15b_instruct", download_if_missing=False)


class AP25Runtime:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lock = threading.RLock()
        self.embedding_model = None
        self.reranker = None
        self.vectordb = None
        self.chunks: list[dict[str, Any]] | None = None
        self.engine: AP25RetrievalEngine | None = None
        self.model_manager = ModelManager(device=self.device)

    def ensure_retrieval_ready(self) -> None:
        with self.lock:
            if self.engine is not None:
                return

            if not PDF_PATH.exists():
                raise FileNotFoundError(f"AP25.pdf not found: {PDF_PATH}")

            EMBEDDING_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            RERANKER_MODEL_DIR.mkdir(parents=True, exist_ok=True)

            ensure_model_downloaded(EMBEDDING_MODEL_NAME, EMBEDDING_MODEL_DIR)
            ensure_model_downloaded(RERANKER_MODEL_NAME, RERANKER_MODEL_DIR)

            chunks = parse_ap25_for_notebook(str(PDF_PATH))
            index_documents = to_langchain_documents(chunks)

            embedding_model = HuggingFaceEmbeddings(
                model_name=str(EMBEDDING_MODEL_DIR),
                model_kwargs={"device": self.device},
                encode_kwargs={"normalize_embeddings": True},
            )

            vectordb = Chroma(
                collection_name=COLLECTION_NAME,
                persist_directory=str(VECTORSTORE_DIR),
                embedding_function=embedding_model,
            )

            if vectordb._collection.count() == 0:
                vectordb = Chroma.from_documents(
                    documents=index_documents,
                    embedding=embedding_model,
                    collection_name=COLLECTION_NAME,
                    persist_directory=str(VECTORSTORE_DIR),
                )

            reranker = CrossEncoder(str(RERANKER_MODEL_DIR), device=self.device)
            engine = AP25RetrievalEngine(chunks=chunks, vectordb=vectordb, reranker=reranker)

            self.embedding_model = embedding_model
            self.vectordb = vectordb
            self.reranker = reranker
            self.chunks = chunks
            self.engine = engine

    def build_context(self, hits: list[dict[str, Any]], max_context_chars: int = MAX_CONTEXT_CHARS) -> str:
        pieces: list[str] = []
        total_chars = 0
        for hit in hits:
            piece = f"{chunk_header(hit)}\n{hit['text']}"
            if total_chars + len(piece) > max_context_chars:
                break
            pieces.append(piece)
            total_chars += len(piece) + 2
        return "\n\n".join(pieces).strip()

    def run_chat(self, messages: list[dict[str, str]], max_new_tokens: int, generation_config: dict[str, Any] | None = None) -> str:
        self.model_manager.ensure_default()
        tokenizer = self.model_manager.tokenizer
        model = self.model_manager.model
        if tokenizer is None or model is None:
            raise RuntimeError("No LLM is loaded")

        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer([prompt_text], return_tensors="pt")
        target_device = model.device if hasattr(model, "device") else torch.device(self.device)
        inputs = {key: value.to(target_device) for key, value in inputs.items()}

        gen_config = dict(generation_config or GENERATION_CONFIG)
        gen_config["max_new_tokens"] = max_new_tokens
        if gen_config.get("pad_token_id") is None:
            gen_config["pad_token_id"] = tokenizer.eos_token_id

        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_config)

        new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def generate_answer(self, query: str, context_hits: list[dict[str, Any]]) -> str:
        context = self.build_context(context_hits)
        user_message = f"Контекст из АП-25:\n{context}\n\nВопрос:\n{query}\n"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        return self.run_chat(
            messages,
            max_new_tokens=GENERATION_CONFIG["max_new_tokens"],
            generation_config=GENERATION_CONFIG,
        )

    def validate_issue_spotting_hit(self, request_text: str, hit: dict[str, Any]) -> dict[str, Any]:
        user_message = f"Кейс:\n{request_text}\n\nПункт АП-25:\n{chunk_header(hit)}\n{hit['text']}\n"
        messages = [
            {"role": "system", "content": ISSUE_SPOTTING_VALIDATOR_PROMPT},
            {"role": "user", "content": user_message},
        ]

        raw_text = self.run_chat(
            messages,
            max_new_tokens=ISSUE_VALIDATOR_CONFIG["max_new_tokens"],
            generation_config=ISSUE_VALIDATOR_CONFIG,
        )
        try:
            parsed = extract_json_object(raw_text)
        except Exception:
            parsed = {
                "decision": "uncertain",
                "applicability_score": 0,
                "confidence": "low",
                "reason": f"Не удалось надежно распарсить ответ валидатора: {raw_text[:200]}",
            }

        decision = coerce_issue_decision(parsed.get("decision", "uncertain"))
        confidence = str(parsed.get("confidence", "low")).strip().lower()
        if confidence not in {"high", "medium", "low"}:
            confidence = "low"

        try:
            applicability_score = int(parsed.get("applicability_score", 0))
        except Exception:
            applicability_score = 0
        applicability_score = max(0, min(applicability_score, 100))

        validated = dict(hit)
        validated["validator_decision"] = decision
        validated["validator_confidence"] = confidence
        validated["applicability_score"] = applicability_score
        validated["validator_reason"] = str(parsed.get("reason", "")).strip()
        validated["validator_raw"] = raw_text
        return validated

    def validate_issue_spotting_hits(self, request_text: str, hits: list[dict[str, Any]], limit: int = 18) -> list[dict[str, Any]]:
        return [self.validate_issue_spotting_hit(request_text, hit) for hit in hits[:limit]]

    def normalized_issue_decision(self, item: dict[str, Any]) -> str:
        decision = item.get("validator_decision", "uncertain")
        rerank_score = safe_score(item.get("rerank_score"))
        paragraph_score = safe_score(item.get("paragraph_score"))
        if decision == "core" and rerank_score < 0.20 and paragraph_score < 1.00:
            return "supporting"
        if decision in {"core", "supporting"} and rerank_score < 0.08 and paragraph_score < 0.75:
            return "uncertain"
        return decision

    def sort_issue_spotting_results(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        def confidence_rank(value: str) -> int:
            return {"high": 0, "medium": 1, "low": 2}.get(value, 2)

        return sorted(
            items,
            key=lambda item: (
                ISSUE_DECISION_ORDER.get(self.normalized_issue_decision(item), 99),
                -safe_score(item.get("paragraph_score")),
                -safe_score(item.get("rerank_score")),
                -int(item.get("applicability_score", 0)),
                confidence_rank(item.get("validator_confidence", "low")),
            ),
        )

    def render_issue_spotting_results(self, items: list[dict[str, Any]], low_confidence: bool = False, limit: int = 15) -> str:
        lines: list[str] = []
        if low_confidence:
            lines.append(
                "Retrieval confidence низкая, поэтому список ниже нужно рассматривать как предварительный shortlist для ручной проверки."
            )
        else:
            lines.append("Ниже приведен shortlist пунктов АП-25 для последующей ручной проверки.")

        grouped = {"core": [], "supporting": [], "uncertain": []}
        for item in self.sort_issue_spotting_results(items)[:limit]:
            grouped.setdefault(self.normalized_issue_decision(item), []).append(item)

        section_titles = {
            "core": "Основные кандидаты",
            "supporting": "Дополнительные кандидаты",
            "uncertain": "Спорные кандидаты",
        }

        for decision in ["core", "supporting", "uncertain"]:
            decision_items = grouped.get(decision, [])
            if not decision_items:
                continue
            lines.append(f"\n{section_titles[decision]}:")
            for item in decision_items:
                excerpt = item.get("text", "").strip().replace("\n", " ")
                excerpt = excerpt[:240].rstrip()
                lines.append(f"- {chunk_header(item)}\n  {excerpt}...")
        return "\n".join(lines).strip()

    def render_issue_spotting_shortlist(self, hits: list[dict[str, Any]], low_confidence: bool = False, limit: int = 15) -> str:
        lines: list[str] = []
        if low_confidence:
            lines.append("Retrieval confidence низкая, поэтому ниже показан только retrieval shortlist без LLM-валидации.")
        else:
            lines.append("Ниже приведен retrieval shortlist пунктов АП-25 для последующей ручной проверки.")
        for item in hits[:limit]:
            excerpt = item.get("text", "").strip().replace("\n", " ")
            excerpt = excerpt[:240].rstrip()
            lines.append(f"- {chunk_header(item)}\n  {excerpt}...")
        return "\n".join(lines).strip()

    def hit_payload(self, hit: dict[str, Any]) -> dict[str, Any]:
        return {
            "chunk_id": hit.get("chunk_id"),
            "paragraph": hit.get("paragraph"),
            "paragraph_label": hit.get("paragraph_label"),
            "paragraph_title": hit.get("paragraph_title"),
            "context": hit.get("context"),
            "clause_markers": hit.get("clause_markers"),
            "text_preview": (hit.get("text", "")[:280] + "...") if len(hit.get("text", "")) > 280 else hit.get("text", ""),
            "rerank_score": safe_score(hit.get("rerank_score")),
            "paragraph_score": safe_score(hit.get("paragraph_score")),
            "fusion_score": safe_score(hit.get("fusion_score")),
            "lexical_score": safe_score(hit.get("lexical_score")),
            "dense_distance": safe_score(hit.get("dense_distance"), default=float("nan")),
            "supplement_reason": hit.get("supplement_reason"),
            "bundle_reason": hit.get("bundle_reason"),
        }

    def query(self, query: str, query_mode: str | None = None, issue_validation_limit: int = 18) -> dict[str, Any]:
        self.ensure_retrieval_ready()
        engine = self.engine
        if engine is None:
            raise RuntimeError("Retrieval engine is not initialized")

        auto_mode = engine.detect_query_mode(query)
        selected_mode = query_mode or auto_mode

        retrieval = engine.retrieve(
            query=query,
            query_mode_override=selected_mode,
            k_dense=TOP_K_DENSE,
            k_lexical=TOP_K_LEXICAL,
            k_fused=TOP_K_FUSED,
            k_final=TOP_K_FINAL,
            max_per_paragraph=MAX_PER_PARAGRAPH,
        )

        hits = retrieval["hits"]
        if not hits:
            return {
                "query": query,
                "query_mode": selected_mode,
                "auto_mode": auto_mode,
                "confidence_label": "low",
                "answer": "В предоставленных параграфах АП-25 нет надежно найденной информации по данному вопросу.",
                "hits": [],
            }

        threshold = retrieval["threshold"]
        composite_score = retrieval["composite_score"]
        visible_hits = hits[: min(6, len(hits))]

        if retrieval["query_mode"] == "issue_spotting":
            if composite_score < threshold:
                answer = self.render_issue_spotting_shortlist(hits, low_confidence=True, limit=issue_validation_limit)
            else:
                validated = self.validate_issue_spotting_hits(query, hits, limit=issue_validation_limit)
                answer = self.render_issue_spotting_results(validated, low_confidence=False, limit=issue_validation_limit)
        else:
            if composite_score < threshold:
                answer = engine.render_candidate_list(hits, low_confidence=True)
            else:
                answer = self.generate_answer(query, visible_hits)

        return {
            "query": query,
            "query_mode": retrieval["query_mode"],
            "auto_mode": auto_mode,
            "threshold": threshold,
            "top_rerank_score": retrieval["top_rerank_score"],
            "composite_score": composite_score,
            "confidence_label": retrieval["confidence_label"],
            "answer": answer,
            "hits": [self.hit_payload(hit) for hit in visible_hits],
            "model": self.model_manager.current_spec.display_name if self.model_manager.current_spec else None,
        }

    def status(self) -> dict[str, Any]:
        current = self.model_manager.current_spec
        return {
            "device": self.device,
            "python": platform.python_version(),
            "retrieval_ready": self.engine is not None,
            "current_model": current.display_name if current else None,
            "current_model_key": current.key if current else None,
            "models": self.model_manager.list_models(),
        }


RUNTIME = AP25Runtime()


class AP25RequestHandler(BaseHTTPRequestHandler):
    server_version = "AP25WebApp/0.1"

    def _send_bytes(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self._send_bytes(status, body, "application/json; charset=utf-8")

    def _send_text(self, text: str, status: int = 200) -> None:
        self._send_bytes(status, text.encode("utf-8"), "text/plain; charset=utf-8")

    def _read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            return {}
        raw = self.rfile.read(content_length)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            return self._serve_static("index.html")
        if parsed.path.startswith("/static/"):
            relative = parsed.path.removeprefix("/static/")
            return self._serve_static(relative)
        if parsed.path == "/api/status":
            return self._send_json({"ok": True, "status": RUNTIME.status()})
        if parsed.path == "/api/models":
            return self._send_json({"ok": True, "models": RUNTIME.model_manager.list_models()})
        self._send_text("Not found", status=404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = self._read_json()
            if parsed.path == "/api/chat":
                query = str(payload.get("query", "")).strip()
                if not query:
                    return self._send_json({"ok": False, "error": "Query is required"}, status=400)
                model_key = payload.get("model_key")
                if model_key:
                    RUNTIME.model_manager.load(str(model_key), download_if_missing=bool(payload.get("download_if_missing", False)))
                result = RUNTIME.query(query=query, query_mode=payload.get("query_mode") or None)
                return self._send_json({"ok": True, "result": result})

            if parsed.path == "/api/models/load":
                model_key = str(payload.get("model_key", "")).strip()
                if not model_key:
                    return self._send_json({"ok": False, "error": "model_key is required"}, status=400)
                result = RUNTIME.model_manager.load(model_key, download_if_missing=bool(payload.get("download_if_missing", False)))
                return self._send_json({"ok": True, "result": result, "status": RUNTIME.status()})

            if parsed.path == "/api/models/unload":
                RUNTIME.model_manager.unload()
                return self._send_json({"ok": True, "status": RUNTIME.status()})

            return self._send_json({"ok": False, "error": "Not found"}, status=404)
        except Exception as exc:
            return self._send_json(
                {
                    "ok": False,
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=3),
                },
                status=500,
            )

    def log_message(self, format: str, *args: Any) -> None:
        message = "%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), format % args)
        sys.stdout.write(message)

    def _serve_static(self, relative_path: str) -> None:
        safe_path = (STATIC_DIR / relative_path).resolve()
        if not str(safe_path).startswith(str(STATIC_DIR.resolve())) or not safe_path.exists() or not safe_path.is_file():
            return self._send_text("Not found", status=404)
        mime_type, _ = mimetypes.guess_type(str(safe_path))
        self._send_bytes(200, safe_path.read_bytes(), mime_type or "application/octet-stream")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AP-25 browser app")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the HTTP server to.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the HTTP server to.")
    parser.add_argument(
        "--preload-default-model",
        action="store_true",
        help="Load the default LLM on startup instead of waiting for the first request.",
    )
    parser.add_argument(
        "--preload-retrieval",
        action="store_true",
        help="Load parser, embeddings, vector DB and reranker on startup.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.preload_retrieval:
        RUNTIME.ensure_retrieval_ready()
    if args.preload_default_model:
        RUNTIME.model_manager.ensure_default()

    server = ThreadingHTTPServer((args.host, args.port), AP25RequestHandler)
    print(f"AP-25 web app started at http://{args.host}:{args.port}")
    print("For LAN access, open the server IP from another computer, for example http://SERVER_IP:8000")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
