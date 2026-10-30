from __future__ import annotations

import csv
import hashlib
import io
import os
import uuid
from math import log2
from pathlib import Path
from statistics import mean
from time import perf_counter

import streamlit as st

from app.config import get_settings
from app.core.vectorstore import RetrievalFilters, RetrievedChunk
from app.services import Services


st.set_page_config(page_title="Rag-Rerank", page_icon="📚", layout="wide")
st.markdown(
    """
    <style>
    .block-container{max-width:1500px;width:100%;min-width:0;padding-top:2rem;padding-bottom:3rem}
    h1{font-size:2.6rem!important;letter-spacing:-.04em}
    [data-testid="stMetric"]{background:#172033!important;border:1px solid #36435d;border-radius:16px;padding:16px;color:#f4f5ff!important}
    [data-testid="stMetric"] *{color:#f4f5ff!important}
    [data-testid="stMetricValue"]{font-size:1.6rem}
    div.stButton>button{border-radius:11px;font-weight:700;min-height:2.7rem}
    section[data-testid="stSidebar"]{background:linear-gradient(180deg,#10172a,#201e48);overflow-x:hidden!important}
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"]{box-sizing:border-box;width:100%;overflow-x:hidden!important;padding-left:1rem!important;padding-right:1rem!important}
    section[data-testid="stSidebar"] .block-container{box-sizing:border-box;width:100%!important;max-width:100%!important;min-width:0;padding-left:.25rem!important;padding-right:.25rem!important}
    section[data-testid="stSidebar"] *{color:#f4f5ff}
    ::selection{background:#635bff;color:#fff}
    </style>
    """,
    unsafe_allow_html=True,
)


def _settings_from_secrets() -> None:
    try:
        for name in (
            "OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_API_BASE", "QDRANT_URL",
            "RE_RANK_ENABLED", "RE_RANK_MODEL_NAME", "RERANK_CANDIDATE_K",
        ):
            if name in st.secrets and st.secrets[name] not in (None, ""):
                os.environ[name] = str(st.secrets[name])
    except (FileNotFoundError, RuntimeError):
        pass


@st.cache_resource
def _services() -> Services:
    _settings_from_secrets()
    get_settings.cache_clear()
    services = Services.build(get_settings())
    services.vectorstore.ensure_collection()
    return services


def _fmt_ms(value: float) -> str:
    return f"{value / 1000:.2f} s"


def _page_metrics(
    ranked: list[RetrievedChunk], relevant_pages: set[int], total_relevant: int, k: int
) -> dict[str, float]:
    hits = [int(chunk.page_number in relevant_pages) for chunk in ranked[:k]]
    relevant_hits = sum(hits)
    precision = relevant_hits / k
    recall = relevant_hits / total_relevant if total_relevant else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    first = next((rank for rank, hit in enumerate(hits, 1) if hit), None)
    dcg = sum(hit / log2(rank + 1) for rank, hit in enumerate(hits, 1))
    ideal_count = min(total_relevant, k)
    idcg = sum(1 / log2(rank + 1) for rank in range(1, ideal_count + 1))
    return {
        "precision_at_k": precision,
        "recall_at_k": recall,
        "f1_at_k": f1,
        "mrr": 1 / first if first else 0.0,
        "ndcg_at_k": dcg / idcg if idcg else 0.0,
    }


def _auc(chunks: list[RetrievedChunk], scores: list[float | None], pages: set[int]) -> float | None:
    pairs = [(chunk.page_number in pages, score) for chunk, score in zip(chunks, scores) if score is not None]
    positives = [score for relevant, score in pairs if relevant]
    negatives = [score for relevant, score in pairs if not relevant]
    if not positives or not negatives:
        return None
    wins = sum(
        1.0 if pos > neg else 0.5 if pos == neg else 0.0
        for pos in positives for neg in negatives
    )
    return wins / (len(positives) * len(negatives))


def _parse_questions(file) -> list[dict[str, object]]:
    content = file.getvalue().decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(content)))
    if not rows or not {"question", "relevant_pages"}.issubset(rows[0]):
        raise ValueError("CSV headers must include question and relevant_pages.")
    if len(rows) > 30:
        raise ValueError("Use no more than 30 questions.")
    parsed = []
    for number, row in enumerate(rows, start=2):
        question = (row.get("question") or "").strip()
        raw_pages = (row.get("relevant_pages") or "").strip()
        try:
            pages = sorted({int(value.strip()) for value in raw_pages.replace("|", ";").split(";") if value.strip()})
        except ValueError as exc:
            raise ValueError(f"Row {number}: page numbers must be integers separated by semicolons.") from exc
        if len(question) < 3 or not pages or min(pages) < 1:
            raise ValueError(f"Row {number}: add a question and its relevant page number(s).")
        parsed.append({"question": question, "relevant_pages": pages})
    return parsed


def _upload_pdf(services: Services, uploaded) -> str:
    data = uploaded.getvalue()
    if len(data) > services.settings.max_upload_size_bytes:
        raise ValueError("PDF exceeds the configured upload-size limit.")
    if not uploaded.name.lower().endswith(".pdf") or not data.startswith(b"%PDF-"):
        raise ValueError("Choose a valid PDF file.")
    fingerprint = hashlib.sha256(data).hexdigest()
    if (
        services.vectorstore.has_document_fingerprint(fingerprint)
        and services.settings.duplicate_upload_policy == "reject"
    ):
        raise ValueError("This PDF is already indexed.")
    doc_id = str(uuid.uuid4())
    path = services.settings.data_dir / "uploads" / f"{doc_id}.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    try:
        services.pipeline.ingest_pdf(doc_id, fingerprint, uploaded.name, path)
    except Exception:
        services.vectorstore.delete_document(doc_id)
        path.unlink(missing_ok=True)
        raise
    return doc_id


def _run_eval(services: Services, doc: dict[str, object], questions: list[dict[str, object]], top_k: int = 5) -> dict[str, object]:
    if not services.settings.re_rank_enabled:
        raise ValueError("Enable RE_RANK_ENABLED=true to compare reranking.")
    services.generation_client.check_configured()
    doc_id = str(doc["doc_id"])
    candidate_k = max(services.settings.rerank_candidate_k, top_k)
    all_chunks = services.vectorstore.list_chunks_for_doc_id(doc_id)
    metric_names = ("precision_at_k", "recall_at_k", "f1_at_k", "mrr", "ndcg_at_k", "roc_auc")
    values = {method: {metric: [] for metric in metric_names} for method in ("dense", "reranked")}
    latency = {method: [] for method in ("dense", "reranked")}
    answers = []

    progress = st.progress(0, text="Preparing evaluation…")
    for number, item in enumerate(questions, start=1):
        question = str(item["question"])
        pages = set(item["relevant_pages"])  # type: ignore[arg-type]
        if any(page > int(doc["page_count"]) for page in pages):
            raise ValueError(f"Question {number}: a relevant page is beyond this PDF's page count.")
        total_relevant = sum(1 for chunk in all_chunks if int(chunk["page_number"]) in pages)
        if not total_relevant:
            raise ValueError(f"Question {number}: the labeled page has no indexed text.")
        filters = RetrievalFilters(doc_id=doc_id)
        dense = services.retriever.retrieve(
            question, top_k=top_k, candidate_k=candidate_k, filters=filters, rerank_enabled=False
        )
        reranked = services.retriever.retrieve(
            question, top_k=top_k, candidate_k=candidate_k, filters=filters, rerank_enabled=True
        )
        metric_pair = {
            "dense": _page_metrics(dense.chunks, pages, total_relevant, top_k),
            "reranked": _page_metrics(reranked.chunks, pages, total_relevant, top_k),
        }
        dense_auc = _auc(dense.candidate_chunks, [chunk.score for chunk in dense.candidate_chunks], pages)
        rerank_auc = _auc(reranked.candidate_chunks, reranked.candidate_reranker_scores or [], pages)
        if dense_auc is not None:
            metric_pair["dense"]["roc_auc"] = dense_auc
        if rerank_auc is not None:
            metric_pair["reranked"]["roc_auc"] = rerank_auc
        dense_ms = dense.embedding_latency_ms + dense.retrieval_latency_ms
        reranked_ms = reranked.embedding_latency_ms + reranked.retrieval_latency_ms + reranked.rerank_latency_ms
        latency["dense"].append(dense_ms)
        latency["reranked"].append(reranked_ms)
        for method in ("dense", "reranked"):
            for metric, value in metric_pair[method].items():
                values[method][metric].append(value)

        prompt = services.prompt_builder.build(question, reranked.chunks)
        answer_started = perf_counter()
        answer = services.generation_client.generate(prompt.prompt).text
        answer_ms = int((perf_counter() - answer_started) * 1000)
        answers.append({
            "question": question,
            "relevant_pages": sorted(pages),
            "dense_mrr": metric_pair["dense"]["mrr"],
            "reranked_mrr": metric_pair["reranked"]["mrr"],
            "dense_latency_ms": dense_ms,
            "reranked_latency_ms": reranked_ms,
            "answer": answer,
            "answer_generation_ms": answer_ms,
        })
        progress.progress(number / len(questions), text=f"Evaluated {number} of {len(questions)} questions…")

    summary = {
        method: {
            metric: mean(scores) if scores else None
            for metric, scores in metrics.items()
        }
        for method, metrics in values.items()
    }
    summary["dense"]["mean_latency_ms"] = mean(latency["dense"])
    summary["reranked"]["mean_latency_ms"] = mean(latency["reranked"])
    return {
        "filename": doc["filename"],
        "question_count": len(answers),
        "candidate_k": candidate_k,
        "top_k": top_k,
        "summary": summary,
        "mean_answer_generation_ms": mean(row["answer_generation_ms"] for row in answers),
        "questions": answers,
    }


def main() -> None:
    st.title("Rag-Rerank")
    st.caption("Ask your PDFs questions and inspect the passages behind each answer.")
    try:
        services = _services()
    except Exception as exc:
        st.error(f"Could not initialize the document system: {exc}")
        st.stop()

    docs = services.vectorstore.list_documents()
    with st.sidebar:
        st.markdown("## Rag-Rerank")
        st.caption("PDF document intelligence")
        st.write("🟢 Vector database")
        st.write("🟢 Answer model" if services.generation_client.is_configured() else "🔴 Configure model key")
        st.write("🟢 Reranker" if services.settings.re_rank_enabled else "🔴 Reranker disabled")
        st.divider()
        st.subheader("Add a PDF")
        upload = st.file_uploader("Choose a PDF", type=["pdf"], key="pdf_upload")
        if upload is not None and st.button("Index PDF", use_container_width=True):
            try:
                with st.spinner("Extracting, embedding, and indexing PDF…"):
                    _upload_pdf(services, upload)
                st.success("PDF indexed.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    tab_qa, tab_retrieval, tab_eval = st.tabs(["Document QA", "Retrieval Results", "Evaluation"])
    with tab_qa:
        st.header("Ask your documents")
        if not docs:
            st.info("Upload and index a text-based PDF in the sidebar to begin.")
        else:
            doc_names = {str(d["filename"]): d for d in docs}
            selected_name = st.selectbox("Search in PDF", list(doc_names))
            top_k = st.select_slider("Passages to use", options=[3, 5, 8, 12, 20], value=5)
            question = st.text_area("Your question", placeholder="Ask a question answered by the selected PDF…", height=110)
            if st.button("Generate answer", type="primary", disabled=not question.strip()):
                try:
                    services.generation_client.check_configured()
                    with st.spinner("Retrieving, reranking, and generating…"):
                        result = services.retriever.retrieve(
                            question.strip(), top_k=top_k,
                            filters=RetrievalFilters(doc_id=str(doc_names[selected_name]["doc_id"])),
                        )
                        prompt = services.prompt_builder.build(question.strip(), result.chunks)
                        answer = services.generation_client.generate(prompt.prompt).text
                    st.session_state["last_qa"] = {
                        "question": question.strip(), "answer": answer,
                        "retrieval": result, "latency_ms": result.embedding_latency_ms + result.retrieval_latency_ms + result.rerank_latency_ms,
                    }
                except Exception as exc:
                    st.error(str(exc))
            last = st.session_state.get("last_qa")
            if last:
                st.subheader("Answer")
                st.write(last["answer"])
                rr = last["retrieval"]
                c1, c2, c3 = st.columns(3)
                c1.metric("Retrieved candidates", rr.original_candidate_count)
                c2.metric("Reranker time", _fmt_ms(rr.rerank_latency_ms))
                c3.metric("Retrieval total", _fmt_ms(last["latency_ms"]))
                st.caption("Source evidence")
                for chunk in rr.chunks:
                    with st.expander(f"{chunk.filename} · page {chunk.page_number} · chunk {chunk.chunk_index}"):
                        st.write(chunk.text)

    with tab_retrieval:
        st.header("Retrieval Results")
        last = st.session_state.get("last_qa")
        if not last:
            st.info("Ask a question first to inspect its retrieved passages.")
        else:
            rr = last["retrieval"]
            st.markdown(f"**Question:** {last['question']}")
            before, after = st.columns(2)
            with before:
                st.subheader("Before reranking")
                for rank, chunk in enumerate(rr.candidate_chunks[:rr.retrieval_config.top_k], 1):
                    st.markdown(f"**#{rank} · page {chunk.page_number} · score {chunk.score:.3f}**")
                    st.write(chunk.text)
                    st.divider()
            with after:
                st.subheader("After reranking")
                if not rr.retrieval_config.reranker_enabled:
                    st.warning("Reranking is disabled.")
                else:
                    for rank, chunk in enumerate(rr.chunks, 1):
                        st.markdown(f"**#{rank} · page {chunk.page_number}**")
                        st.write(chunk.text)
                        st.divider()

    with tab_eval:
        st.header("Evaluate a PDF")
        st.caption("Compare dense retrieval and reranking using checked page labels, then inspect the generated reranked answer for each question.")
        if not docs:
            st.info("Upload and index a PDF before evaluating it.")
        else:
            doc_names = {str(d["filename"]): d for d in docs}
            eval_name = st.selectbox("PDF to evaluate", list(doc_names), key="eval_pdf")
            template = "question,relevant_pages\n\"Who is the project guide?\",2\n"
            st.download_button("Download CSV example", template, "evaluation-questions.csv", "text/csv")
            csv_file = st.file_uploader("Questions CSV (question,relevant_pages)", type=["csv"], key="eval_csv")
            st.caption("One answer-generation API call is made per question. Retrieval metric labels treat chunks from the listed pages as relevant.")
            if csv_file and st.button("Run evaluation", type="primary"):
                try:
                    questions = _parse_questions(csv_file)
                    with st.spinner("Evaluating both retrieval methods and generating answers…"):
                        data = _run_eval(services, doc_names[eval_name], questions)
                    st.session_state["last_eval"] = data
                except Exception as exc:
                    st.error(str(exc))
            data = st.session_state.get("last_eval")
            if data:
                st.markdown(f"### Results for {data['filename']} · {data['question_count']} questions")
                summary = data["summary"]
                score_labels = [
                    ("precision_at_k", "Precision@5"), ("recall_at_k", "Recall@5"),
                    ("f1_at_k", "F1@5"), ("mrr", "MRR"),
                    ("ndcg_at_k", "nDCG@5"), ("roc_auc", "AUC-ROC"),
                ]
                cols = st.columns(4)
                for i, (key, label) in enumerate(score_labels):
                    dense, rerank = summary["dense"][key], summary["reranked"][key]
                    cols[i % 4].metric(label, "—" if rerank is None else f"{rerank:.1%}", delta=None if dense is None or rerank is None else f"{(rerank-dense)*100:+.1f} pts")
                st.subheader("Metrics comparison")
                chart_rows = [
                    {"Metric": label, "Dense retrieval": summary["dense"][key], "With reranking": summary["reranked"][key]}
                    for key, label in score_labels if summary["dense"][key] is not None and summary["reranked"][key] is not None
                ]
                if chart_rows:
                    st.bar_chart(chart_rows, x="Metric", y=["Dense retrieval", "With reranking"], color=["#635bff", "#079669"])
                latency_rows = [{"Method": "Dense retrieval", "Seconds": summary["dense"]["mean_latency_ms"] / 1000}, {"Method": "With reranking", "Seconds": summary["reranked"]["mean_latency_ms"] / 1000}]
                st.subheader("Mean retrieval latency (answer generation excluded)")
                st.bar_chart(latency_rows, x="Method", y="Seconds", color="#635bff")
                st.metric("Mean answer generation time", _fmt_ms(float(data["mean_answer_generation_ms"])))
                for row in data["questions"]:
                    with st.expander(f"{row['question']} · relevant page(s): {', '.join(map(str,row['relevant_pages']))}"):
                        left, right = st.columns(2)
                        left.metric("Dense MRR", f"{row['dense_mrr']:.1%}")
                        right.metric("Reranked MRR", f"{row['reranked_mrr']:.1%}")
                        st.markdown("**Generated answer (reranked passages)**")
                        st.write(row["answer"])
                        st.caption(f"Answer generation: {_fmt_ms(row['answer_generation_ms'])}")


if __name__ == "__main__":
    main()
