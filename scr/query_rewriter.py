import re
from typing import Any


class QueryRewriter:
    def __init__(self, llm: Any | None = None):
        self.llm = llm

    def rewrite(self, query: str) -> str:
        normalized = self._normalize(query)
        if not normalized:
            return query.strip()

        if self.llm is None:
            return normalized

        prompt = (
            "You are a query rewriting assistant.\n"
            "Rewrite the user's query into a clear, standalone search query suitable "
            "for retrieval embeddings.\n"
            "Rules:\n"
            "- Preserve meaning and key details.\n"
            "- Resolve pronouns only if the reference is explicit in the query.\n"
            "- Do not add new facts or assumptions.\n"
            "- Keep it concise and focused on the information need.\n"
            "Return only the rewritten query, nothing else.\n\n"
            f"User query: {normalized}"
        )
        try:
            response = self.llm.invoke(prompt)
        except Exception:
            return normalized

        rewritten = getattr(response, "content", "") if response is not None else ""
        cleaned = self._clean_response(rewritten)
        return cleaned if cleaned else normalized

    def _normalize(self, query: str) -> str:
        compact = re.sub(r"\s+", " ", query).strip()
        return compact

    def _clean_response(self, text: str) -> str:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            return ""
        if cleaned.startswith(("'", '"')) and cleaned.endswith(("'", '"')):
            cleaned = cleaned[1:-1].strip()
        return cleaned
