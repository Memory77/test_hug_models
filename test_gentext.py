from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

#Facteurs Affectant la Qualité des Résultats
#Qualité et Quantité des Données d'Entraînement :

#La qualité des données utilisées pour le fine-tuning joue un rôle crucial. Si les données contiennent des erreurs ou ne sont pas représentatives de l'utilisation prévue, les résultats seront médiocres.
#La quantité de données est également importante. Un petit jeu de données peut ne pas être suffisant pour bien ajuster le modèle.
#Hyperparamètres de l'Entraînement :

#Le nombre d'époques (num_train_epochs), la taille du batch (per_device_train_batch_size), et le taux d'apprentissage sont des hyperparamètres critiques. Des valeurs mal ajustées peuvent conduire à un sur-apprentissage (overfitting) ou à un sous-apprentissage (underfitting).
#Architecture du Modèle :

#La taille et la complexité du modèle utilisé peuvent également influencer les résultats. DistilGPT-2 est un modèle léger et peut ne pas capturer autant de nuances qu'un modèle plus grand comme GPT-3.
#Charger le modèle et le tokenizer fine-tuné

model_path = './fine_tuned_model'
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)


# Fonction pour générer du texte
def generate_text(prompt, model, tokenizer, max_length=50):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

# Exemple de prompt
prompt = "Qui est Henoch?"
generated_text = generate_text(prompt, model, tokenizer)
print(f"Answer: {generated_text}")