from langchain_core.runnables import RunnableLambda

from src.config import get_retriever
from src.chains import (
    contextualize_chain,
    document_answer_chain,
    intent_chain,
    history_answer_chain,
    general_chain,
)

NO_ANSWER = "I don't know based on the provided documents."
OUT_OF_SCOPE_ANSWER = (
    "Sorry, I can only answer questions about the company documents "
    "(for example leave, working hours, or data policies)."
)


def detect_intent(question, history):
    raw = intent_chain.invoke({"input": question, "chat_history": history}).strip().upper()
    # Tolerate outputs like "DOCUMENT." or "Intent: HISTORY"
    for label in ("OUT_OF_SCOPE", "DOCUMENT", "HISTORY", "GENERAL"):
        if label in raw:
            return label
    return "GENERAL"


def retrieve_documents(question, history):
    # 1. Rewrite the question into a standalone question
    standalone_question = contextualize_chain.invoke(
        {"input": question, "chat_history": history}
    )

    # 2. Retrieve relevant chunks
    documents = get_retriever().invoke(standalone_question)
    context = "\n\n".join(doc.page_content for doc in documents)

    sources = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata["page"] + 1 if "page" in doc.metadata else None,
            "content": doc.page_content,
        }
        for doc in documents
    ]

    # 3. Handle empty retrieval
    if not context.strip():
        return NO_ANSWER, standalone_question, sources

    # 4. Generate the document-based answer
    answer = document_answer_chain.invoke(
        {"input": standalone_question, "context": context}
    )
    return answer, standalone_question, sources


def ask(question, history):
    """Answer a question and return details for the UI.

    Returns a dict: {answer, intent, rewritten_question, sources}.
    """
    intent = detect_intent(question, history)
    rewritten, sources = None, []

    if intent == "OUT_OF_SCOPE":
        answer = OUT_OF_SCOPE_ANSWER
    elif intent == "DOCUMENT":
        answer, rewritten, sources = retrieve_documents(question, history)
    elif intent == "HISTORY":
        answer = history_answer_chain.invoke({"input": question, "chat_history": history})
    else:
        answer = general_chain.invoke({"input": question, "chat_history": history})

    return {
        "answer": answer,
        "intent": intent,
        "rewritten_question": rewritten,
        "sources": sources,
    }


def route_request(inputs):
    result = ask(inputs["input"], inputs["chat_history"])
    print("\nDetected Intent:", result["intent"])
    if result["rewritten_question"]:
        print("Rewritten Question:", result["rewritten_question"])
    return result["answer"]


rag_chain = RunnableLambda(route_request)
