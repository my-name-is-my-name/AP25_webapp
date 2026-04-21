from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any


LIST_QUERY_PATTERNS = [
    r"какие\s+пункт",
    r"какие\s+параграф",
    r"что\s+нужно\s+проанализ",
    r"что\s+проанализ",
    r"при\s+запросе\s+на\s+ремонт",
    r"изменени[ея]\s+пассажировместим",
    r"модификац",
    r"ремонт",
]

TARGETED_COMPLIANCE_PATTERNS = [
    r"на\s+какие\s+нагруз",
    r"на\s+что\s+оцениват",
    r"на\s+что\s+проверят",
    r"по\s+каким\s+нагруз",
    r"какие\s+расчетные\s+услов",
    r"какие\s+расчетные\s+перегруз",
    r"какие\s+нагрузки\s+учитыват",
]

CLAUSE_REF_RE = re.compile(
    r"пункт(?:а|е|у|ом|ов|ах)?\s+\(([A-Za-zА-Яа-я])\)(?:\(\d+\))?",
    flags=re.IGNORECASE,
)
PARAGRAPH_REF_RE = re.compile(r"(?:§\s*)?(25\.\d+[A-Za-zА-Яа-я]?)", flags=re.IGNORECASE)
TOKEN_RE = re.compile(r"[a-zа-я0-9]+", flags=re.IGNORECASE)


def normalize_text(text: str) -> str:
    return text.lower().replace("ё", "е")


def normalize_clause_ref(marker: str) -> str:
    mapping = {
        "а": "a",
        "б": "b",
        "в": "v",
        "г": "g",
        "д": "d",
        "е": "e",
        "ж": "zh",
        "з": "z",
        "и": "i",
        "й": "j",
        "к": "k",
        "л": "l",
        "м": "m",
        "н": "n",
        "о": "o",
        "п": "p",
        "р": "r",
        "с": "c",
        "т": "t",
        "у": "u",
        "ф": "f",
        "х": "h",
    }
    value = normalize_text(marker.strip())
    return "".join(mapping.get(char, char) for char in value)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize_text(text))


def make_ngrams(tokens: list[str], min_n: int = 2, max_n: int = 3) -> set[str]:
    ngrams: set[str] = set()
    for n in range(min_n, max_n + 1):
        if len(tokens) < n:
            continue
        for start in range(len(tokens) - n + 1):
            ngrams.add(" ".join(tokens[start : start + n]))
    return ngrams


def extract_paragraph_refs(text: str) -> set[str]:
    refs = set()
    for match in PARAGRAPH_REF_RE.findall(text):
        refs.add(normalize_text(match))
    return refs


def safe_score(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(numeric) or math.isinf(numeric):
        return default
    return numeric


def markers_to_text(chunk: dict[str, Any]) -> str:
    if chunk.get("clause_markers"):
        return ", ".join(chunk["clause_markers"])
    if chunk.get("clause_marker"):
        return chunk["clause_marker"]
    return ""


def chunk_header(chunk: dict[str, Any]) -> str:
    parts = [
        chunk["paragraph_label"],
        chunk.get("paragraph_title", ""),
        chunk.get("context", ""),
    ]
    markers = markers_to_text(chunk)
    if markers:
        parts.append(f"markers: {markers}")
    return " | ".join(part for part in parts if part)


class AP25RetrievalEngine:
    VALID_QUERY_MODES = {
        "normative_lookup",
        "issue_spotting",
    }
    QUERY_MODE_ALIASES = {
        "fact_lookup": "normative_lookup",
        "targeted_compliance_lookup": "normative_lookup",
    }

    def __init__(self, chunks: list[dict[str, Any]], vectordb: Any, reranker: Any):
        self.chunks = chunks
        self.vectordb = vectordb
        self.reranker = reranker
        self.chunk_lookup = {chunk["chunk_id"]: chunk for chunk in chunks}
        self.paragraph_chunks: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.paragraph_clause_lookup: dict[str, dict[str, dict[str, Any]]] = {}
        self.doc_freq: Counter[str] = Counter()
        self.doc_count = max(len(chunks), 1)

        for chunk in chunks:
            self.paragraph_chunks[chunk["paragraph"]].append(chunk)

        for paragraph, items in self.paragraph_chunks.items():
            clause_map: dict[str, dict[str, Any]] = {}
            for item in items:
                normalized = item.get("normalized_clause_marker")
                if not normalized and item.get("clause_marker"):
                    normalized = normalize_clause_ref(item["clause_marker"].strip("()"))
                if normalized and normalized not in clause_map:
                    clause_map[normalized] = item
            self.paragraph_clause_lookup[paragraph] = clause_map

        for chunk in chunks:
            doc_tokens = set(
                tokenize(chunk.get("paragraph_title", ""))
                + tokenize(chunk.get("context", ""))
                + tokenize(chunk.get("text", ""))
            )
            self.doc_freq.update(doc_tokens)

    def detect_query_mode(self, query: str) -> str:
        normalized = normalize_text(query)
        for pattern in LIST_QUERY_PATTERNS:
            if re.search(pattern, normalized):
                return "issue_spotting"
        for pattern in TARGETED_COMPLIANCE_PATTERNS:
            if re.search(pattern, normalized):
                return "normative_lookup"
        return "normative_lookup"

    def confidence_threshold_for_mode(self, mode: str) -> float:
        if mode == "issue_spotting":
            return 0.40
        return 0.47

    def format_embedding_query(self, query: str) -> str:
        return f"query: {query.strip()}"

    def build_rerank_text(self, chunk: dict[str, Any]) -> str:
        return f"{chunk_header(chunk)}\n{chunk['text']}"

    def query_terms(self, query: str) -> list[str]:
        terms = []
        for token in tokenize(query):
            if len(token) < 3:
                continue
            # dynamic downweighting replaces hard-coded stopwords
            df_ratio = self.doc_freq.get(token, 0) / self.doc_count
            # domain terms like "расчетные" and "эксплуатационные" remain important
            # in AP-25 even when they are frequent across the corpus.
            if df_ratio >= 0.80 and not token.startswith(("расчет", "эксплуатац")):
                continue
            terms.append(token)
        return terms

    def token_weight(self, token: str) -> float:
        df = self.doc_freq.get(token, 0)
        return math.log((1 + self.doc_count) / (1 + df)) + 1.0

    def lexical_score(self, chunk: dict[str, Any], query_terms: list[str]) -> float:
        if not query_terms:
            return 0.0

        title_text = chunk.get("paragraph_title", "")
        context_text = chunk.get("context", "")
        body_text = chunk.get("text", "")
        title_tokens = set(tokenize(title_text))
        context_tokens = set(tokenize(context_text))
        body_tokens = set(tokenize(body_text))
        marker_tokens = set(tokenize(markers_to_text(chunk)))
        all_tokens = title_tokens | context_tokens | body_tokens | marker_tokens

        score = 0.0
        for token in query_terms:
            weight = self.token_weight(token)
            if token in title_tokens:
                score += 4.0 * weight
            if token in context_tokens:
                score += 3.0 * weight
            if token in marker_tokens:
                score += 1.0 * weight
            if token in body_tokens:
                score += 1.5 * weight

        unique_terms = list(dict.fromkeys(query_terms))
        matched_terms = [token for token in unique_terms if token in all_tokens]
        matched_count = len(matched_terms)
        coverage_ratio = matched_count / max(len(unique_terms), 1)
        score += coverage_ratio * 6.0

        # Penalize chunks that only match one broad term from the query.
        if len(unique_terms) >= 2 and matched_count < 2:
            score *= 0.55

        weighted_terms = sorted(unique_terms, key=self.token_weight, reverse=True)
        top_terms = weighted_terms[: min(3, len(weighted_terms))]
        if top_terms and all(token not in all_tokens for token in top_terms):
            score *= 0.35

        query_token_ngrams = make_ngrams(unique_terms, min_n=2, max_n=3)
        title_ngrams = make_ngrams(tokenize(title_text), min_n=2, max_n=3)
        context_ngrams = make_ngrams(tokenize(context_text), min_n=2, max_n=3)
        body_ngrams = make_ngrams(tokenize(body_text)[:80], min_n=2, max_n=3)

        for ngram in query_token_ngrams:
            if ngram in title_ngrams:
                score += 7.0
            elif ngram in context_ngrams:
                score += 4.0
            elif ngram in body_ngrams:
                score += 2.0

        normalized_query = normalize_text(" ".join(unique_terms))
        normalized_title = normalize_text(title_text)
        if normalized_query and normalized_title:
            if normalized_query in normalized_title:
                score += 10.0
            elif normalized_title and normalized_title in normalized_query:
                score += 8.0

        query_refs = extract_paragraph_refs(" ".join(unique_terms))
        if query_refs and normalize_text(chunk["paragraph"]) in query_refs:
            score += 20.0

        # boost cohesive structural matches rather than injected synonyms
        query_text = normalize_text(" ".join(query_terms))
        if "аварийн" in query_text and "посад" in query_text:
            if {"аварийной", "посадки"} & context_tokens:
                score += 4.0
        if "крес" in query_text and ("кресла" in title_tokens or "кресла" in body_tokens):
            score += 2.0
        if "шасс" in query_text and ("шасси" in context_tokens or "шасси" in body_tokens):
            score += 2.0

        return score

    def retrieve_dense(self, query: str, k: int = 40) -> list[dict[str, Any]]:
        results = self.vectordb.similarity_search_with_score(self.format_embedding_query(query), k=k)
        hits = []
        seen_chunk_ids = set()

        for rank, (doc, distance) in enumerate(results, start=1):
            chunk_id = doc.metadata.get("chunk_id")
            if not chunk_id or chunk_id in seen_chunk_ids or chunk_id not in self.chunk_lookup:
                continue
            seen_chunk_ids.add(chunk_id)
            chunk = dict(self.chunk_lookup[chunk_id])
            chunk["dense_distance"] = float(distance)
            chunk["dense_rank"] = rank
            hits.append(chunk)

        return hits

    def retrieve_lexical(self, query: str, k: int = 60) -> list[dict[str, Any]]:
        terms = self.query_terms(query)
        scored_hits = []

        for chunk in self.chunks:
            score = self.lexical_score(chunk, terms)
            if score <= 0:
                continue
            item = dict(chunk)
            item["lexical_score"] = float(score)
            scored_hits.append(item)

        scored_hits.sort(key=lambda item: item["lexical_score"], reverse=True)
        for rank, item in enumerate(scored_hits, start=1):
            item["lexical_rank"] = rank

        return scored_hits[:k]

    def fuse_ranked_hits(self, *ranked_lists: list[dict[str, Any]], limit: int = 60, rrf_k: int = 60) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        fusion_scores: dict[str, float] = {}

        for hits in ranked_lists:
            for rank, hit in enumerate(hits, start=1):
                chunk_id = hit["chunk_id"]
                if chunk_id not in merged:
                    merged[chunk_id] = dict(hit)
                else:
                    for key, value in hit.items():
                        merged[chunk_id].setdefault(key, value)
                fusion_scores[chunk_id] = fusion_scores.get(chunk_id, 0.0) + (1.0 / (rrf_k + rank))

        fused = []
        for chunk_id, hit in merged.items():
            item = dict(hit)
            item["fusion_score"] = fusion_scores[chunk_id]
            fused.append(item)

        fused.sort(key=lambda item: item["fusion_score"], reverse=True)
        return fused[:limit]

    def rerank_hits(self, query: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not hits:
            return []

        pairs = [[query, self.build_rerank_text(hit)] for hit in hits]
        scores = self.reranker.predict(pairs, batch_size=16, show_progress_bar=False)

        reranked = []
        for hit, score in zip(hits, scores):
            item = dict(hit)
            item["rerank_score"] = float(score)
            reranked.append(item)

        reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
        return reranked

    def paragraph_scores(self, hits: list[dict[str, Any]]) -> dict[str, float]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for hit in hits:
            grouped[hit["paragraph"]].append(hit)

        scores: dict[str, float] = {}
        for paragraph, items in grouped.items():
            rerank_values = sorted((safe_score(item.get("rerank_score")) for item in items), reverse=True)
            lexical_values = sorted((safe_score(item.get("lexical_score")) for item in items), reverse=True)
            fusion_values = sorted((safe_score(item.get("fusion_score")) for item in items), reverse=True)
            score = 0.0
            if rerank_values:
                score += rerank_values[0]
            if len(rerank_values) > 1:
                score += 0.20 * rerank_values[1]
            if lexical_values:
                score += 0.05 * lexical_values[0]
            if fusion_values:
                score += 0.05 * fusion_values[0]
            scores[paragraph] = score
        return scores

    def is_numeric_chunk(self, chunk: dict[str, Any]) -> bool:
        return bool(re.search(r"\b\d+(?:[.,]\d+)?\s*g\b|\b\d+(?:[.,]\d+)?\b", chunk.get("text", ""), flags=re.IGNORECASE))

    def composite_confidence(self, mode: str, hits: list[dict[str, Any]]) -> dict[str, Any]:
        if not hits:
            return {
                "top_rerank_score": 0.0,
                "composite_score": 0.0,
                "confidence_label": "low",
                "threshold": self.confidence_threshold_for_mode(mode),
            }

        sorted_hits = sorted(hits, key=lambda item: safe_score(item.get("rerank_score")), reverse=True)
        top1 = safe_score(sorted_hits[0].get("rerank_score"))
        top2 = safe_score(sorted_hits[1].get("rerank_score")) if len(sorted_hits) > 1 else 0.0
        gap = max(top1 - top2, 0.0)

        top_paragraph = sorted_hits[0]["paragraph"]
        same_paragraph_count = sum(1 for hit in sorted_hits[:5] if hit["paragraph"] == top_paragraph)
        supplement_present = any(bool(hit.get("supplement_reason")) for hit in hits)
        lexical_signal = max((safe_score(hit.get("lexical_score")) for hit in hits), default=0.0)
        numeric_present = any(self.is_numeric_chunk(hit) for hit in hits[:4])
        title_context_match = any(
            hit.get("paragraph_title") or hit.get("context") for hit in hits[:3]
        )
        title_overlap_present = any(
            safe_score(hit.get("lexical_score")) >= 20.0 and hit.get("paragraph_title")
            for hit in hits[:4]
        )

        composite = top1
        composite += min(gap, 0.20) * 0.5
        composite += min(same_paragraph_count / 5.0, 0.6) * 0.10
        if supplement_present:
            composite += 0.08
        if numeric_present and mode == "normative_lookup":
            composite += 0.06
        if title_context_match:
            composite += 0.03
        if title_overlap_present:
            composite += 0.04
        composite += min(lexical_signal / 100.0, 0.10)

        threshold = self.confidence_threshold_for_mode(mode)
        if composite >= threshold + 0.12:
            label = "high"
        elif composite >= threshold:
            label = "medium"
        else:
            label = "low"

        return {
            "top_rerank_score": top1,
            "top2_rerank_score": top2,
            "gap_score": gap,
            "same_paragraph_count": same_paragraph_count,
            "supplement_present": supplement_present,
            "numeric_present": numeric_present,
            "composite_score": composite,
            "confidence_label": label,
            "threshold": threshold,
        }

    def build_paragraph_bundle(
        self,
        paragraph: str,
        reranked_hits: list[dict[str, Any]],
        max_hits: int = 4,
        include_intro: bool = True,
    ) -> list[dict[str, Any]]:
        reranked_lookup = {hit["chunk_id"]: dict(hit) for hit in reranked_hits}
        paragraph_hits = [dict(hit) for hit in reranked_hits if hit["paragraph"] == paragraph]
        paragraph_hits.sort(key=lambda item: safe_score(item.get("rerank_score")), reverse=True)
        selected: list[dict[str, Any]] = []
        selected_ids = set()

        if include_intro:
            intro_candidates = []
            for chunk in self.paragraph_chunks.get(paragraph, []):
                marker = chunk.get("normalized_clause_marker")
                intro_priority = 0
                if marker in {"a", "b"}:
                    intro_priority = 2
                elif not marker:
                    intro_priority = 1
                if intro_priority:
                    candidate = dict(reranked_lookup.get(chunk["chunk_id"], chunk))
                    candidate["bundle_reason"] = "intro_clause"
                    candidate["_bundle_priority"] = intro_priority
                    intro_candidates.append(candidate)
            intro_candidates.sort(
                key=lambda item: (item["_bundle_priority"], safe_score(item.get("rerank_score"))),
                reverse=True,
            )
            for candidate in intro_candidates[:2]:
                candidate.pop("_bundle_priority", None)
                if candidate["chunk_id"] not in selected_ids:
                    selected.append(candidate)
                    selected_ids.add(candidate["chunk_id"])

        for hit in paragraph_hits:
            if hit["chunk_id"] in selected_ids:
                continue
            hit["bundle_reason"] = "top_paragraph_evidence"
            selected.append(hit)
            selected_ids.add(hit["chunk_id"])
            if len(selected) >= max_hits:
                break

        selected.sort(key=lambda item: safe_score(item.get("rerank_score")), reverse=True)
        return selected[:max_hits]

    def select_hits(
        self,
        reranked_hits: list[dict[str, Any]],
        mode: str,
        limit: int = 8,
        max_per_paragraph: int = 2,
    ) -> list[dict[str, Any]]:
        if not reranked_hits:
            return []

        paragraph_score_map = self.paragraph_scores(reranked_hits)
        selected: list[dict[str, Any]] = []
        selected_ids = set()

        if mode == "normative_lookup":
            paragraph_order = sorted(paragraph_score_map, key=paragraph_score_map.get, reverse=True)
            if paragraph_order:
                primary = paragraph_order[0]
                bundle_size = 4
                for hit in self.build_paragraph_bundle(primary, reranked_hits, max_hits=bundle_size, include_intro=True):
                    if hit["chunk_id"] in selected_ids:
                        continue
                    hit = dict(hit)
                    hit["paragraph_score"] = paragraph_score_map[primary]
                    selected.append(hit)
                    selected_ids.add(hit["chunk_id"])
                    if len(selected) >= limit:
                        return selected

            for paragraph in paragraph_order[1:]:
                paragraph_hits = [hit for hit in reranked_hits if hit["paragraph"] == paragraph]
                paragraph_hits.sort(key=lambda item: safe_score(item.get("rerank_score")), reverse=True)
                for hit in paragraph_hits[:1]:
                    if hit["chunk_id"] in selected_ids:
                        continue
                    hit = dict(hit)
                    hit["paragraph_score"] = paragraph_score_map[paragraph]
                    selected.append(hit)
                    selected_ids.add(hit["chunk_id"])
                    if len(selected) >= limit:
                        return selected
        else:
            effective_limit = max(limit, 30) if mode == "issue_spotting" else limit
            effective_max_per_paragraph = max(max_per_paragraph, 3) if mode == "issue_spotting" else max_per_paragraph
            paragraph_counts: dict[str, int] = {}
            for hit in reranked_hits:
                paragraph = hit["paragraph"]
                if paragraph_counts.get(paragraph, 0) >= effective_max_per_paragraph:
                    continue
                hit = dict(hit)
                hit["paragraph_score"] = paragraph_score_map.get(paragraph, 0.0)
                selected.append(hit)
                selected_ids.add(hit["chunk_id"])
                paragraph_counts[paragraph] = paragraph_counts.get(paragraph, 0) + 1
                if len(selected) >= effective_limit:
                    break

        return selected

    def resolve_intra_paragraph_refs(self, chunk: dict[str, Any]) -> list[dict[str, Any]]:
        clause_map = self.paragraph_clause_lookup.get(chunk["paragraph"], {})
        resolved = []
        seen_normalized = set()

        for raw_marker in CLAUSE_REF_RE.findall(chunk.get("text", "")):
            normalized = normalize_clause_ref(raw_marker)
            if not normalized or normalized in seen_normalized:
                continue
            sibling = clause_map.get(normalized)
            if sibling and sibling["chunk_id"] != chunk["chunk_id"]:
                resolved.append(sibling)
                seen_normalized.add(normalized)

        return resolved

    def supplement_intra_paragraph_refs(self, hits: list[dict[str, Any]], max_extra_chunks: int = 3) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        seen_chunk_ids = set()
        extra_chunks = 0
        hit_lookup = {hit["chunk_id"]: hit for hit in hits}

        for hit in hits:
            if hit["chunk_id"] not in seen_chunk_ids:
                selected.append(hit)
                seen_chunk_ids.add(hit["chunk_id"])

            for ref_chunk in self.resolve_intra_paragraph_refs(hit):
                if ref_chunk["chunk_id"] in seen_chunk_ids or extra_chunks >= max_extra_chunks:
                    continue
                extra = dict(hit_lookup.get(ref_chunk["chunk_id"], ref_chunk))
                extra["supplement_reason"] = f"same_paragraph_ref_from:{hit['chunk_id']}"
                extra.setdefault("paragraph_score", hit.get("paragraph_score", 0.0))
                selected.append(extra)
                seen_chunk_ids.add(extra["chunk_id"])
                extra_chunks += 1

        return selected

    def retrieve(
        self,
        query: str,
        query_mode_override: str | None = None,
        k_dense: int = 40,
        k_lexical: int = 60,
        k_fused: int = 60,
        k_final: int = 8,
        max_per_paragraph: int = 2,
    ) -> dict[str, Any]:
        mode = self.QUERY_MODE_ALIASES.get(query_mode_override or "", query_mode_override or self.detect_query_mode(query))
        if mode not in self.VALID_QUERY_MODES:
            raise ValueError(f"Unsupported query mode: {mode}")
        dense_hits = self.retrieve_dense(query, k=k_dense)
        lexical_hits = self.retrieve_lexical(query, k=k_lexical)
        fused_hits = self.fuse_ranked_hits(dense_hits, lexical_hits, limit=k_fused)
        reranked_hits = self.rerank_hits(query, fused_hits)

        if not reranked_hits:
            return {
                "query_mode": mode,
                "threshold": self.confidence_threshold_for_mode(mode),
                "top_rerank_score": 0.0,
                "composite_score": 0.0,
                "confidence_label": "low",
                "hits": [],
            }

        selected_hits = self.select_hits(
            reranked_hits=reranked_hits,
            mode=mode,
            limit=k_final,
            max_per_paragraph=max_per_paragraph,
        )
        supplemented_hits = self.supplement_intra_paragraph_refs(selected_hits)
        confidence = self.composite_confidence(mode, supplemented_hits)

        return {
            "query_mode": mode,
            "threshold": confidence["threshold"],
            "top_rerank_score": confidence["top_rerank_score"],
            "composite_score": confidence["composite_score"],
            "confidence_label": confidence["confidence_label"],
            "top2_rerank_score": confidence.get("top2_rerank_score", 0.0),
            "gap_score": confidence.get("gap_score", 0.0),
            "same_paragraph_count": confidence.get("same_paragraph_count", 0),
            "supplement_present": confidence.get("supplement_present", False),
            "hits": supplemented_hits,
        }

    @staticmethod
    def hit_confidence_label(hit: dict[str, Any]) -> str:
        rerank_score = safe_score(hit.get("rerank_score"))
        paragraph_score = safe_score(hit.get("paragraph_score"))
        if rerank_score >= 0.75 or paragraph_score >= 0.90:
            return "высокая"
        if rerank_score >= 0.45 or paragraph_score >= 0.60:
            return "средняя"
        return "низкая"

    @staticmethod
    def hit_reason(hit: dict[str, Any]) -> str:
        reasons = []
        if hit.get("bundle_reason") == "intro_clause":
            reasons.append("входит в опорный пакет основного параграфа")
        elif hit.get("bundle_reason") == "top_paragraph_evidence":
            reasons.append("сильный подпункт внутри ведущего параграфа")
        if hit.get("supplement_reason"):
            reasons.append("подтянут по внутренней ссылке того же параграфа")
        if safe_score(hit.get("paragraph_score")) >= 0.75:
            reasons.append("параграф в целом выглядит сильным кандидатом")
        if not reasons:
            reasons.append("совпал по заголовку, контексту или тексту нормы")
        return "; ".join(reasons)

    @classmethod
    def render_candidate_list(cls, hits: list[dict[str, Any]], low_confidence: bool = False, limit: int = 5) -> str:
        lines = []
        if low_confidence:
            lines.append("Надежного ответа по АП-25 не найдено. Ниже только ближайшие кандидаты для ручной проверки:")
        else:
            lines.append("Ниже приведены наиболее релевантные пункты для последующего анализа:")

        for hit in hits[:limit]:
            snippet = hit["text"][:220].strip()
            lines.append(
                f"- {chunk_header(hit)}\n"
                f"  Почему попал: {cls.hit_reason(hit)}.\n"
                f"  Уверенность: {cls.hit_confidence_label(hit)}.\n"
                f"  Фрагмент: {snippet}"
            )

        return "\n".join(lines)
