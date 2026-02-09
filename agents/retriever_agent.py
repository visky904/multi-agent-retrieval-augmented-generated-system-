import ollama

def retrieve(context, query):
    prompt = f"""
You are a retrieval agent.
Extract only relevant facts from context.

Context:
{context}

Question: {query}
"""
    return ollama.chat(model="mistral", messages=[{"role":"user","content":prompt}])["message"]["content"]
