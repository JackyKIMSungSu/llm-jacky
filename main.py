import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent / "src"))
load_dotenv()


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python main.py "<주제>"')
        sys.exit(1)

    from llm_jacky.research import run_research

    topic = " ".join(sys.argv[1:])
    result = run_research(topic)

    print(f"\n=== 주제 ===\n{result.topic}\n")
    print(f"=== 요약 ===\n{result.summary}\n")
    print("=== 출처 ===")
    for i, s in enumerate(result.sources, 1):
        print(f"[{i}] {s.title}\n    {s.url}")


if __name__ == "__main__":
    main()
