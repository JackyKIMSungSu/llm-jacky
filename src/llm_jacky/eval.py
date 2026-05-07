"""LangSmith evaluation: dataset + 6 metrics for the full pipeline.

Metrics (plan-0 7단계):
1. grounded         — 본문 주장이 수집 자료로 뒷받침되는지 (LLM judge)
2. citations        — `[n]` 인용 마커 존재 비율 (programmatic)
3. seo_title        — 제목 길이/유효성 + 주제 관련성 (mix)
4. structure        — H1/H2/길이 (programmatic)
5. foreigner        — 외국인 독자 이해도 (LLM judge)
6. brand_tone       — 브랜드 톤 일관성 (LLM judge, rubric 가변)
"""

from __future__ import annotations

import re
from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from .rag import index_research
from .research import run_research
from .seo import generate_seo
from .writing import write_draft

DEFAULT_DATASET_NAME = "llm-jacky-blog-eval"
DEFAULT_JUDGE_MODEL = "gpt-4o-mini"

SEED_TOPICS: list[str] = [
    "외국인을 위한 서울 길거리 음식 가이드",
    "한국 봄꽃 명소 추천",
    "K-드라마 촬영지 투어 (서울)",
    "한국 전통차 카페 추천",
    "외국인이 좋아하는 한국 디저트",
]

BRAND_RUBRIC = """\
- 외국인(영어권/비한국어권) 독자가 한국 문화를 이해하기 쉽도록 친근한 어조
- 구체적인 장소·시간·감각 묘사 (불필요한 미사여구 지양)
- 인용 출처를 [n] 형식으로 명확히
- 과장·검증 안 된 단정 금지
"""


# ---------- Target ----------

def target(inputs: dict[str, Any]) -> dict[str, Any]:
    """Run the full pipeline once. Used as LangSmith eval target."""
    topic = inputs["topic"]
    research = run_research(topic, max_results=3)
    index_research(research)
    draft = write_draft(topic, k=3)
    seo = generate_seo(topic, draft.draft)
    return {
        "draft": draft.draft,
        "sources": draft.sources,
        "seo": seo.model_dump(),
    }


# ---------- LLM judge plumbing ----------

class _JudgeScore(BaseModel):
    score: float = Field(description="0.0(나쁨)~1.0(좋음) 사이 점수.", ge=0.0, le=1.0)
    reason: str = Field(description="2~3 문장 한국어 근거.")


def _judge(system: str, user: str, model: str = DEFAULT_JUDGE_MODEL) -> _JudgeScore:
    llm = ChatOpenAI(model=model, temperature=0).with_structured_output(_JudgeScore)
    return llm.invoke([("system", system), ("human", user)])  # type: ignore[return-value]


# ---------- Evaluators ----------

def evaluate_grounded(outputs: dict, **_: Any) -> dict:
    draft = outputs["draft"]
    sources = outputs["sources"]
    sources_block = "\n".join(f"[{i+1}] {s.get('title','')} — {s.get('url','')}"
                              for i, s in enumerate(sources))
    judge = _judge(
        system="당신은 사실성 평가자입니다. 초안 본문의 주장들이 수집한 출처로 뒷받침되는지 평가합니다.",
        user=(
            f"수집 출처:\n{sources_block}\n\n"
            f"초안:\n{draft}\n\n"
            "본문 주장 중 출처와 모순되거나 출처에 없는 내용이 얼마나 적은지 0~1로 채점."
        ),
    )
    return {"key": "grounded", "score": judge.score, "comment": judge.reason}


_CITATION_PATTERN = re.compile(r"\[\d+\]")


def evaluate_citations(outputs: dict, **_: Any) -> dict:
    draft: str = outputs["draft"]
    body = re.sub(r"^#.*$", "", draft, flags=re.MULTILINE)
    paragraphs = [p.strip() for p in body.split("\n\n") if len(p.strip()) > 30]
    if not paragraphs:
        return {"key": "citations", "score": 0.0, "comment": "본문 단락이 없음"}
    cited = sum(1 for p in paragraphs if _CITATION_PATTERN.search(p))
    score = cited / len(paragraphs)
    return {
        "key": "citations",
        "score": score,
        "comment": f"인용 단락 {cited}/{len(paragraphs)}",
    }


def evaluate_seo_title(inputs: dict, outputs: dict, **_: Any) -> dict:
    seo = outputs["seo"]
    title: str = seo.get("title", "")
    topic: str = inputs["topic"]
    length_ok = 1.0 if 1 <= len(title) <= 60 else 0.0
    judge = _judge(
        system="당신은 SEO 편집자입니다. 제목이 주제를 정확히 반영하고 클릭 매력도 있는지 평가합니다.",
        user=f"주제: {topic}\n제목: {title}\n0~1로 채점.",
    )
    score = 0.4 * length_ok + 0.6 * judge.score
    return {
        "key": "seo_title",
        "score": score,
        "comment": f"len={len(title)} ok={length_ok} judge={judge.score:.2f} — {judge.reason}",
    }


def evaluate_structure(outputs: dict, **_: Any) -> dict:
    draft: str = outputs["draft"]
    has_h1 = bool(re.search(r"^#\s+\S", draft, flags=re.MULTILINE))
    h2_count = len(re.findall(r"^##\s+\S", draft, flags=re.MULTILINE))
    length = len(draft)
    h2_ok = 1.0 if 3 <= h2_count <= 6 else (0.5 if h2_count > 0 else 0.0)
    h1_ok = 1.0 if has_h1 else 0.0
    len_ok = 1.0 if 800 <= length <= 4000 else (0.5 if length >= 400 else 0.0)
    score = (h1_ok + h2_ok + len_ok) / 3
    return {
        "key": "structure",
        "score": score,
        "comment": f"h1={has_h1} h2={h2_count} len={length}",
    }


def evaluate_foreigner(outputs: dict, **_: Any) -> dict:
    draft = outputs["draft"]
    judge = _judge(
        system="당신은 외국인(비한국어권) 독자 관점의 평가자입니다. 한국 문화 배경 지식이 적어도 글이 이해되는지 봅니다.",
        user=(
            f"초안:\n{draft}\n\n"
            "어려운 고유명사/관용 표현에 충분한 설명이 있는지, 흐름이 명확한지 0~1로 채점."
        ),
    )
    return {"key": "foreigner", "score": judge.score, "comment": judge.reason}


def evaluate_brand_tone(outputs: dict, **_: Any) -> dict:
    draft = outputs["draft"]
    judge = _judge(
        system=(
            "당신은 브랜드 톤 가이드 검수자입니다. 다음 루브릭에 부합하는 정도를 평가합니다.\n"
            f"{BRAND_RUBRIC}"
        ),
        user=f"초안:\n{draft}\n\n루브릭 부합도 0~1.",
    )
    return {"key": "brand_tone", "score": judge.score, "comment": judge.reason}


ALL_EVALUATORS = [
    evaluate_grounded,
    evaluate_citations,
    evaluate_seo_title,
    evaluate_structure,
    evaluate_foreigner,
    evaluate_brand_tone,
]


# ---------- Dataset + entry ----------

def upsert_dataset(client, name: str = DEFAULT_DATASET_NAME, topics: list[str] | None = None) -> str:
    """Create the dataset if missing, then ensure each topic exists as an example."""
    topics = topics or SEED_TOPICS
    try:
        ds = client.read_dataset(dataset_name=name)
    except Exception:
        ds = client.create_dataset(dataset_name=name, description="llm-jacky 풀 파이프라인 평가용")

    existing = {e.inputs.get("topic") for e in client.list_examples(dataset_id=ds.id)}
    new = [t for t in topics if t not in existing]
    if new:
        client.create_examples(
            inputs=[{"topic": t} for t in new],
            dataset_id=ds.id,
        )
    return ds.id


def run_evaluation(
    *,
    topics: list[str] | None = None,
    dataset_name: str = DEFAULT_DATASET_NAME,
    experiment_prefix: str = "full-pipeline",
    max_concurrency: int = 1,
):
    """Upsert dataset and run Client.evaluate over all evaluators."""
    from langsmith import Client

    client = Client()
    upsert_dataset(client, dataset_name, topics)
    return client.evaluate(
        target,
        data=dataset_name,
        evaluators=ALL_EVALUATORS,
        experiment_prefix=experiment_prefix,
        max_concurrency=max_concurrency,
    )
