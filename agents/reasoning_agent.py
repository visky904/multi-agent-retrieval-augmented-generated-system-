import ollama

def reason(retrieved_info):
    prompt = f"""
You are a reasoning agent.
Analyze logically and derive insights.

Information:
{retrieved_info}
"""
    return ollama.chat(model="llama3", messages=[{"role":"user","content":prompt}])["message"]["content"]
