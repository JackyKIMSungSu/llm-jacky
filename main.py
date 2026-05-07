"""Full pipeline CLI: topic → research → RAG → draft → SEO → (optional) WordPress."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent / "src"))
load_dotenv()


def _save_draft(slug: str, title: str, description: str, tags: list[str], body: str) -> Path:
    drafts_dir = Path("data/drafts")
    drafts_dir.mkdir(parents=True, exist_ok=True)
    path = drafts_dir / f"{slug}.md"
    front_matter = (
        "---\n"
        f"title: {title}\n"
        f"description: {description}\n"
        f"tags: {', '.join(tags)}\n"
        f"slug: {slug}\n"
        "---\n\n"
    )
    path.write_text(front_matter + body, encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="topic → research → RAG → draft → SEO → (optional) WordPress publish"
    )
    parser.add_argument("topic", help="블로그 글 주제")
    parser.add_argument("--publish", action="store_true", help="WordPress 에 업로드")
    parser.add_argument(
        "--status",
        default="draft",
        choices=["draft", "publish"],
        help="WP 게시 상태 (기본 draft — 사람 검토 후 승인 정책)",
    )
    parser.add_argument("--max-results", type=int, default=5, help="Tavily 검색 결과 수")
    parser.add_argument("--k", type=int, default=5, help="RAG retriever k")
    parser.add_argument("--no-save", action="store_true", help="data/drafts/ 저장 생략")
    args = parser.parse_args()

    from llm_jacky.publish import publish_post
    from llm_jacky.rag import index_research
    from llm_jacky.research import run_research
    from llm_jacky.seo import generate_seo
    from llm_jacky.writing import write_draft

    print(f"[1/5] research: {args.topic!r}")
    research = run_research(args.topic, max_results=args.max_results)
    print(f"      sources={len(research.sources)} summary={len(research.summary)}자")

    print("[2/5] index → Chroma")
    chunks = index_research(research)
    print(f"      chunks={chunks}")

    print("[3/5] write draft (Claude)")
    draft = write_draft(args.topic, k=args.k)
    print(f"      draft={len(draft.draft)}자 sources={len(draft.sources)}")

    print("[4/5] SEO meta (OpenAI)")
    seo = generate_seo(args.topic, draft.draft)
    print(f"      title={seo.title!r}")
    print(f"      slug={seo.slug!r}")
    print(f"      tags={seo.tags}")

    if not args.no_save:
        path = _save_draft(seo.slug, seo.title, seo.meta_description, seo.tags, draft.draft)
        print(f"      saved={path}")

    if args.publish:
        print(f"[5/5] publish to WordPress (status={args.status})")
        result = publish_post(
            title=seo.title,
            content_md=draft.draft,
            excerpt=seo.meta_description,
            slug=seo.slug,
            tags=seo.tags,
            status=args.status,
        )
        print(f"      id={result.id} status={result.status}")
        print(f"      edit={result.edit_url}")
    else:
        print("[5/5] skip publish — `--publish` 로 활성화")


if __name__ == "__main__":
    main()
