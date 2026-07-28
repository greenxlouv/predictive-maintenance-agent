"""
정비 매뉴얼 8개 .md 파일을 ChromaDB에 인덱싱하는 1회성 스크립트.

이 스크립트는 LangGraph 노드가 아니라, Agent 3를 실행하기 전에 딱 한 번
(또는 매뉴얼 내용이 바뀔 때마다) 실행해두는 준비 스크립트다.

임베딩: OpenAI 대신 로컬 한국어 임베딩 모델(jhgan/ko-sroberta-multitask)
사용. .env에 OPENAI_API_KEY가 없으므로(팀은 anthropic/gemini/ollama만
씀), API 키 없이 로컬에서 도는 HuggingFace 임베딩으로 대체했다. 최초
실행 시 모델(약 400MB)을 자동 다운로드한다.

역할 분리
---------
정확한 숫자/스펙(부품, 소요시간, 임계값 등)은 manual_data.py에 코드
상수로 이미 옮겨뒀음. 여기서 인덱싱하는 건 그 숫자들의 "배경 설명·근거"
를 검색하기 위한 용도다. 정확한 수치 자체는 이 RAG 결과에 의존하지
않는다 (agent3_planner.py 참고).

사용법
------
    pip install langchain-huggingface sentence-transformers langchain-chroma --break-system-packages
    python index_manuals.py --manuals-dir ./manuals --persist-dir ./chroma_db

사전 조건: manuals-dir 안에 아래 8개 파일이 그대로의 이름으로 있어야 함
(파일명 맨 앞 숫자를 section 메타데이터로 사용).
    1_위험대응.md ... 8_소요시간.md
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

EMBEDDING_MODEL_NAME = "jhgan/ko-sroberta-multitask"


def parse_section_number(filename: str) -> int:
    match = re.match(r"^(\d+)_", filename)
    if not match:
        raise ValueError(f"파일명에서 section 번호를 못 찾음: {filename}")
    return int(match.group(1))


def load_manual_documents(manuals_dir: Path):
    from langchain_core.documents import Document

    docs = []
    for path in sorted(manuals_dir.glob("*.md")):
        section = parse_section_number(path.name)
        text = path.read_text(encoding="utf-8")
        docs.append(
            Document(
                page_content=text,
                metadata={"section": section, "source": path.name},
            )
        )
    return docs


def get_embeddings():
    """API 키 필요 없는 로컬 한국어 임베딩 모델."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def build_index(manuals_dir: Path, persist_dir: Path, collection_name: str) -> None:
    from langchain_chroma import Chroma

    docs = load_manual_documents(manuals_dir)
    if not docs:
        raise FileNotFoundError(f"{manuals_dir}에서 .md 파일을 찾지 못함")

    print(f"{len(docs)}개 매뉴얼 문서 로드됨:")
    for d in docs:
        print(f"  - section {d.metadata['section']}: {d.metadata['source']}")

    print(f"\n임베딩 모델 로딩 중: {EMBEDDING_MODEL_NAME} (최초 1회 다운로드, 시간 걸릴 수 있음)")
    embeddings = get_embeddings()

    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(persist_dir),
    )
    print(f"\n인덱싱 완료 → {persist_dir} (collection: {collection_name})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="정비 매뉴얼을 ChromaDB에 인덱싱")
    parser.add_argument("--manuals-dir", type=Path, default=Path("./manuals"))
    parser.add_argument("--persist-dir", type=Path, default=Path("./chroma_db"))
    parser.add_argument("--collection-name", type=str, default="maintenance_manuals")
    args = parser.parse_args()

    build_index(args.manuals_dir, args.persist_dir, args.collection_name)
