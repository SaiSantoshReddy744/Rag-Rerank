from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from app.core.generation_client import GenerationClient
from app.core.prompt_builder import PromptBuilder
from app.core.reranker import Reranker
from app.core.retrieval import Retriever
from app.core.vectorstore import VectorStore
from app.ingestion.embedder import Embedder
from app.ingestion.pipeline import IngestionPipeline


@dataclass
class Services:
    settings: Settings
    embedder: Embedder
    vectorstore: VectorStore
    retriever: Retriever
    prompt_builder: PromptBuilder
    generation_client: GenerationClient
    pipeline: IngestionPipeline

    @classmethod
    def build(cls, settings: Settings) -> Services:
        embedder = Embedder(settings)
        vectorstore = VectorStore(settings)
        retriever = Retriever(settings, embedder, vectorstore, reranker=Reranker(settings))
        prompt_builder = PromptBuilder(settings)
        generation_client = GenerationClient(settings)
        pipeline = IngestionPipeline(settings, vectorstore, embedder)
        return cls(
            settings=settings,
            embedder=embedder,
            vectorstore=vectorstore,
            retriever=retriever,
            prompt_builder=prompt_builder,
            generation_client=generation_client,
            pipeline=pipeline,
        )
