from transformers import pipeline

# utiliser une base de connaissances de cette manière peut être moins coûteux que d'effectuer un fine-tuning complet 
# #d'un modèle de langage. Voici pourquoi :

#Pas de Coût d'Entraînement :

#Le fine-tuning d'un modèle de langage nécessite une puissance de calcul considérable, souvent sur des GPU ou des TPU, 
# ce qui peut être coûteux. Utiliser un modèle pré-entraîné avec une base de connaissances n'implique pas de réentraîner
# le modèle, ce qui réduit ces coûts.

#Moins de Ressources :
#Fournir un contexte pertinent au modèle pour chaque requête nécessite moins de ressources que l'entraînement complet du modèle. 
# Vous pouvez exécuter ces opérations sur des machines moins puissantes.
#Flexibilité et Rapidité :

#Vous pouvez facilement mettre à jour ou étendre votre base de connaissances sans devoir réentraîner le modèle. 
# Ajouter de nouvelles informations ou documents est rapide et ne demande pas beaucoup de ressources.

from transformers import pipeline

# Utiliser un pipeline de question-réponse
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Exemple de base de connaissances (texte simple)
with open("henoch.txt", "r", encoding="utf-8") as f:
    documents = f.read()

# Fonction pour répondre aux questions en utilisant le modèle de question-réponse
def answer_question(question, context):
    result = qa_pipeline(question=question, context=context)
    return result['answer']

# Exemple de question
question = "Qui est Enoch?"
answer = answer_question(question, documents)
print(f"Question sur base connaissance : {question}")
print(f"Answer: {answer}")