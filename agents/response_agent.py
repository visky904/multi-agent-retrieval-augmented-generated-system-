import ollama

def respond(reasoned_text):
    prompt = f"""
You are a response agent.
Generate a concise, clear final answer.

Analysis:
{reasoned_text}
"""
    return ollama.chat(model="phi", messages=[{"role":"user","content":prompt}])["message"]["content"]
