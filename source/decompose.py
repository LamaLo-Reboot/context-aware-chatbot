import os
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def decompose_query(query):
    
    #décompose une question complexe en sous-questions.

    prompt = f"""
Tu es un assistant spécialisé en décomposition de requêtes.
Ton objectif : transformer une question complexe en sous-tâches simples et indépendantes.

Règles :
- Ne garde que les sous-questions nécessaires pour répondre à la demande.
- Entre 1 et 5 sous-tâches.
- Retourne uniquement la liste des sous-questions, une par ligne.

Question de l'utilisateur :
"{query}"

Sous questions :
"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.choices[0].message.content

    subtasks = [line.strip("-• ").strip() for line in text.split("\n") if line.strip()]
    
    return subtasks