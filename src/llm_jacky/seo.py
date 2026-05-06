"""SEO chain: draft -> OpenAI -> title / meta description / tags / slug."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langsmith import traceable
from pydantic import BaseModel, Field

DEFAULT_MODEL = "gpt-4o-mini"


class SeoMeta(BaseModel):
    title: str = Field(description="60자 이하의 SEO 친화 제목. 과장 금지.")
    meta_description: str = Field(
        description="150~160자 메타 디스크립션. 핵심 베네핏을 명확히 담는다."
    )
    tags: list[str] = Field(
        description="5~8개 태그. 한국어 또는 영어 키워드. 중복 금지."
    )
    slug: str = Field(
        description="URL 슬러그. 영문 소문자와 하이픈만, 50자 이하."
    )


SEO_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 한국 문화 블로그의 SEO 편집자입니다. "
            "주어진 초안에 있는 사실만 사용해 SEO 메타데이터를 생성합니다. "
            "초안에 없는 정보는 추가하지 않습니다.",
        ),
        (
            "human",
            "주제: {topic}\n\n"
            "블로그 초안:\n{draft}\n\n"
            "위 초안을 바탕으로 SEO 메타데이터를 생성하세요.",
        ),
    ]
)


@traceable(name="seo", run_type="chain")
def generate_seo(
    topic: str,
    draft: str,
    *,
    model: str = DEFAULT_MODEL,
) -> SeoMeta:
    llm = ChatOpenAI(model=model, temperature=0)
    chain = SEO_PROMPT | llm.with_structured_output(SeoMeta)
    return chain.invoke({"topic": topic, "draft": draft})
