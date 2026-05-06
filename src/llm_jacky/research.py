"""Research chain: topic -> web search -> grounded summary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


@dataclass
class Source:
    title: str
    url: str
    content: str


@dataclass
class ResearchResult:
    topic: str
    sources: list[Source]
    summary: str


SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 블로그 작성용 리서치 어시스턴트입니다. "
            "주어진 검색 결과만을 근거로 주제에 대한 핵심 사실과 인용 가능한 포인트를 "
            "한국어로 정리합니다. 검색 결과에 없는 정보는 절대 만들지 않습니다.",
        ),
        (
            "human",
            "주제: {topic}\n\n"
            "검색 결과:\n{results}\n\n"
            "다음 형식으로 정리하세요.\n"
            "1) 핵심 요약 (3~5문장)\n"
            "2) 주요 사실/포인트 (불릿)\n"
            "3) 출처별 한 줄 요약 ([n] 제목 - URL)",
        ),
    ]
)


def _format_results(results: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for i, r in enumerate(results, 1):
        blocks.append(
            f"[{i}] {r.get('title', '')}\n"
            f"URL: {r.get('url', '')}\n"
            f"{r.get('content', '')}"
        )
    return "\n\n".join(blocks)


def run_research(
    topic: str,
    *,
    max_results: int = 5,
    model: str = "gpt-4o-mini",
) -> ResearchResult:
    search = TavilySearchResults(max_results=max_results)
    raw: list[dict[str, Any]] = search.invoke(topic)

    sources = [
        Source(
            title=r.get("title", ""),
            url=r.get("url", ""),
            content=r.get("content", ""),
        )
        for r in raw
    ]

    llm = ChatOpenAI(model=model, temperature=0)
    chain = SUMMARY_PROMPT | llm | StrOutputParser()
    summary = chain.invoke({"topic": topic, "results": _format_results(raw)})

    return ResearchResult(topic=topic, sources=sources, summary=summary)
