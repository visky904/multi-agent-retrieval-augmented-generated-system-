from vector_store import load_db
from agents.retriever_agent import retrieve
from agents.reasoning_agent import reason
from agents.response_agent import respond

db = load_db()

def multi_agent_rag(query):
    docs = db.similarity_search(query, k=22)
    context = "\n".join([doc.page_content for doc in docs])

    retrieved = retrieve(context, query)
    reasoning = reason(retrieved)
    final = respond(reasoning)

    return final

while True:
    q = input("\nAsk: ")
    print("\nAnswer:", multi_agent_rag(q))
