"""Writing chain: RAG retrieval -> Claude blog draft."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable

from llm_jacky.rag import DEFAULT_COLLECTION, DEFAULT_PERSIST_DIR, get_retriever

DEFAULT_MODEL = "claude-sonnet-4-6"


@dataclass
class DraftResult:
    topic: str
    draft: str
    sources: list[dict]


DRAFT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 외국인 독자를 위한 한국 문화 블로그의 작가입니다. "
            "독자는 한국 문화 배경 지식이 적은 비한국어권 사람이라고 가정하세요. "
            "주어진 검색 자료(컨텍스트)에 있는 사실만 사용해 한국어 블로그 초안을 작성합니다. "
            "컨텍스트에 없는 정보는 절대 만들지 마세요. "
            "출처가 필요한 주장 뒤에는 [n] 형식으로 참조 번호를 붙입니다.\n"
            "외국인 독자 친화 규칙:\n"
            "- 고유명사(지명/음식명/인명/축제명 등)가 처음 등장할 때 한 줄 이내 부연 설명을 곁들입니다 "
            "(예: '광장시장(서울 종로의 100년 된 재래시장)'). 컨텍스트에 설명이 없으면 위치·종류 정도만 짧게.\n"
            "- 한국식 줄임말·관용 표현은 풀어 씁니다.\n"
            "- 영문 표기가 도움이 되는 핵심 단어는 괄호로 병기 (예: '한복(hanbok)').",
        ),
        (
            "human",
            "주제: {topic}\n\n"
            "참고 자료 (컨텍스트):\n{context}\n\n"
            "다음 구조의 마크다운 블로그 초안을 작성하세요.\n"
            "- 제목 (H1)\n"
            "- 들어가며 (2~3문장)\n"
            "- 본문 섹션 3~5개 (H2 + 설명)\n"
            "- 마무리\n"
            "- 참조: 각 [n] 에 해당하는 제목과 URL\n\n"
            "분량은 800~1200자 내외.",
        ),
    ]
)


def _format_context(docs: list[Document]) -> str:
    blocks: list[str] = []
    for i, d in enumerate(docs, 1):
        meta = d.metadata
        blocks.append(
            f"[{i}] {meta.get('title', '')}\n"
            f"URL: {meta.get('url', '')}\n"
            f"{d.page_content}"
        )
    return "\n\n".join(blocks)


def _dedupe_sources(docs: list[Document]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for d in docs:
        url = d.metadata.get("url")
        if not url or url in seen:
            continue
        seen.add(url)
        out.append({"title": d.metadata.get("title", ""), "url": url})
    return out


@traceable(name="writing", run_type="chain")
def write_draft(
    topic: str,
    *,
    k: int = 5,
    model: str = DEFAULT_MODEL,
    persist_dir: str | Path = DEFAULT_PERSIST_DIR,
    collection: str = DEFAULT_COLLECTION,
) -> DraftResult:
    retriever = get_retriever(persist_dir=persist_dir, collection=collection, k=k)
    docs = retriever.invoke(topic)

    llm = ChatAnthropic(model=model, temperature=0, max_tokens=2000)
    chain = DRAFT_PROMPT | llm | StrOutputParser()
    draft = chain.invoke({"topic": topic, "context": _format_context(docs)})

    return DraftResult(topic=topic, draft=draft, sources=_dedupe_sources(docs))
