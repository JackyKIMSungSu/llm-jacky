"""RAG store: chunk + embed ResearchResult sources into Chroma."""

from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from llm_jacky.research import ResearchResult

DEFAULT_PERSIST_DIR = "data/chroma"
DEFAULT_COLLECTION = "research"
EMBEDDING_MODEL = "text-embedding-3-small"


def _vector_store(
    persist_dir: str | Path = DEFAULT_PERSIST_DIR,
    collection: str = DEFAULT_COLLECTION,
) -> Chroma:
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=collection,
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=str(persist_dir),
    )


def _to_documents(result: ResearchResult) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs: list[Document] = []
    for src_idx, src in enumerate(result.sources):
        if not src.content:
            continue
        for chunk_idx, chunk in enumerate(splitter.split_text(src.content)):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "topic": result.topic,
                        "title": src.title,
                        "url": src.url,
                        "source_index": src_idx,
                        "chunk_index": chunk_idx,
                    },
                )
            )
    return docs


def _doc_id(doc: Document) -> str:
    return f"{doc.metadata['url']}#{doc.metadata['chunk_index']}"


def index_research(
    result: ResearchResult,
    *,
    persist_dir: str | Path = DEFAULT_PERSIST_DIR,
    collection: str = DEFAULT_COLLECTION,
) -> int:
    """Chunk, embed, and upsert sources. Returns number of chunks stored."""
    docs = _to_documents(result)
    if not docs:
        return 0
    vs = _vector_store(persist_dir, collection)
    vs.add_documents(docs, ids=[_doc_id(d) for d in docs])
    return len(docs)


def get_retriever(
    *,
    persist_dir: str | Path = DEFAULT_PERSIST_DIR,
    collection: str = DEFAULT_COLLECTION,
    k: int = 4,
):
    """Return a LangChain retriever over the indexed corpus."""
    return _vector_store(persist_dir, collection).as_retriever(search_kwargs={"k": k})
