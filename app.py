"""Streamlit UI for the llm-jacky pipeline.

Pages:
- Pipeline: topic → research → draft → SEO → (review) → WordPress publish
- Evaluation: 시드 토픽으로 6 메트릭 평가, LangSmith 실험 링크
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
load_dotenv(PROJECT_ROOT / ".env")

st.set_page_config(page_title="llm-jacky", layout="wide")


def _env_status() -> dict[str, bool]:
    return {
        "OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
        "ANTHROPIC_API_KEY": bool(os.getenv("ANTHROPIC_API_KEY")),
        "TAVILY_API_KEY": bool(os.getenv("TAVILY_API_KEY")),
        "LANGSMITH_API_KEY": bool(os.getenv("LANGSMITH_API_KEY")),
        "WORDPRESS_URL": bool(os.getenv("WORDPRESS_URL")),
        "WORDPRESS_USERNAME": bool(os.getenv("WORDPRESS_USERNAME")),
        "WORDPRESS_APP_PASSWORD": bool(os.getenv("WORDPRESS_APP_PASSWORD")),
    }


def _sidebar() -> str:
    st.sidebar.title("llm-jacky")
    page = st.sidebar.radio("Page", ["Pipeline", "Evaluation"], label_visibility="collapsed")
    st.sidebar.divider()
    st.sidebar.caption("환경 변수")
    for k, ok in _env_status().items():
        st.sidebar.write(("✅ " if ok else "❌ ") + k)
    return page


def _page_pipeline() -> None:
    from llm_jacky.publish import publish_post
    from llm_jacky.rag import index_research
    from llm_jacky.research import run_research
    from llm_jacky.seo import generate_seo
    from llm_jacky.writing import write_draft

    st.title("Blog Pipeline")
    st.caption("주제를 입력하면 research → RAG → Claude 초안 → SEO 까지 실행. 검토 후 WordPress 발행.")

    for k in ("research", "draft", "seo", "publish_result"):
        st.session_state.setdefault(k, None)

    with st.form("pipeline-form"):
        topic = st.text_input("주제", placeholder="예) 외국인을 위한 서울 길거리 음식 가이드")
        col1, col2 = st.columns(2)
        max_results = col1.slider("Tavily max_results", 1, 10, 5)
        k = col2.slider("RAG retriever k", 1, 10, 5)
        submitted = st.form_submit_button("초안 생성")

    if submitted and topic.strip():
        # Reset publish state on new run.
        st.session_state.publish_result = None
        with st.status("실행 중...", expanded=True) as status:
            status.write("[1/4] Tavily 검색 + 요약")
            research = run_research(topic, max_results=max_results)
            status.write(f"  sources={len(research.sources)} summary={len(research.summary)}자")

            status.write("[2/4] Chroma 인덱싱")
            chunks = index_research(research)
            status.write(f"  chunks={chunks}")

            status.write("[3/4] Claude 초안 작성")
            draft = write_draft(topic, k=k)
            status.write(f"  draft={len(draft.draft)}자 sources={len(draft.sources)}")

            status.write("[4/4] OpenAI SEO 메타")
            seo = generate_seo(topic, draft.draft)
            status.write(f"  title={seo.title} | slug={seo.slug}")

            status.update(label="초안 준비 완료", state="complete")

        st.session_state.research = research
        st.session_state.draft = draft
        st.session_state.seo = seo

    if st.session_state.draft is None:
        return

    research = st.session_state.research
    draft = st.session_state.draft
    seo = st.session_state.seo

    st.divider()
    tab_draft, tab_seo, tab_sources = st.tabs(["📝 초안", "🔎 SEO", "📚 출처"])

    with tab_draft:
        st.markdown(draft.draft)

    with tab_seo:
        st.write(f"**Title** ({len(seo.title)}자)")
        st.code(seo.title, language=None)
        st.write(f"**Slug**")
        st.code(seo.slug, language=None)
        st.write(f"**Meta Description** ({len(seo.meta_description)}자)")
        st.code(seo.meta_description, language=None)
        st.write("**Tags**")
        st.write(seo.tags)

    with tab_sources:
        for i, s in enumerate(research.sources, 1):
            st.markdown(f"**[{i}] [{s.title}]({s.url})**")
            st.caption(s.content[:300] + ("..." if len(s.content) > 300 else ""))

    st.divider()
    st.subheader("WordPress 발행")
    if not _env_status()["WORDPRESS_URL"]:
        st.warning("WORDPRESS_URL 등 .env 가 비어 있어 발행 비활성. README 의 셋업 절차 참고.")
        return

    col1, col2 = st.columns([1, 3])
    status_choice = col1.radio("상태", ["draft", "publish"], horizontal=False, index=0)
    col2.caption("기본 draft — plan-0 의 '사람 검토 후 승인' 정책. publish 선택 시 즉시 게시.")

    if st.button("WordPress 에 업로드", type="primary"):
        with st.spinner("uploading..."):
            result = publish_post(
                title=seo.title,
                content_md=draft.draft,
                excerpt=seo.meta_description,
                slug=seo.slug,
                tags=seo.tags,
                status=status_choice,
            )
        st.session_state.publish_result = result

    if st.session_state.publish_result is not None:
        r = st.session_state.publish_result
        st.success(f"업로드 완료 — id={r.id}, status={r.status}")
        st.markdown(f"[관리자 페이지에서 편집]({r.edit_url})")
        if r.status == "publish":
            st.markdown(f"[게시된 글 보기]({r.url})")


def _page_evaluation() -> None:
    from llm_jacky.eval import SEED_TOPICS, run_evaluation

    st.title("Evaluation")
    st.caption("LangSmith 데이터셋 `llm-jacky-blog-eval` 의 시드 토픽으로 6 메트릭 자동 평가.")

    with st.expander("시드 토픽", expanded=False):
        for t in SEED_TOPICS:
            st.write("- ", t)

    col1, col2, col3 = st.columns([2, 1, 1])
    prefix = col1.text_input("실험 prefix", value="ui-run")
    concurrency = col2.slider("동시성", 1, 4, 2)
    n_topics = col3.slider("토픽 수", 1, len(SEED_TOPICS), len(SEED_TOPICS))

    st.caption(
        f"비용 추정: {n_topics} × 풀 파이프라인 + {n_topics * 4} LLM judge. "
        "1 토픽당 ~25~45초."
    )

    if st.button("평가 실행", type="primary"):
        topics = SEED_TOPICS[:n_topics]
        with st.status(f"{len(topics)}개 토픽 평가 중...", expanded=True) as status:
            status.write("LangSmith 데이터셋 업서트 + Client.evaluate 호출")
            results = run_evaluation(
                topics=topics,
                experiment_prefix=prefix,
                max_concurrency=concurrency,
            )
            status.update(label=f"완료 — {results.experiment_name}", state="complete")
        st.session_state.eval_results = results

    if st.session_state.get("eval_results") is None:
        return

    results = st.session_state.eval_results

    st.divider()
    st.subheader(f"실험: `{results.experiment_name}`")

    rows = []
    by_metric = defaultdict(list)
    for r in results:
        run = r.get("run")
        topic = run.inputs.get("topic") if run else "?"
        scores = {ev.key: ev.score for ev in r.get("evaluation_results", {}).get("results", [])}
        rows.append({"topic": topic, **scores})
        for k, v in scores.items():
            by_metric[k].append(v)

    if rows:
        df = pd.DataFrame(rows)
        avg_row = {"topic": "AVG", **{k: sum(v) / len(v) for k, v in by_metric.items()}}
        df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
        st.dataframe(
            df.style.format({c: "{:.2f}" for c in df.columns if c != "topic"}),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown(
        "[LangSmith 실험 페이지에서 자세히 보기](https://smith.langchain.com/) "
        "(좌측 사이드바 → Datasets → `llm-jacky-blog-eval` → Experiments)"
    )


def main() -> None:
    page = _sidebar()
    if page == "Pipeline":
        _page_pipeline()
    else:
        _page_evaluation()


main()
