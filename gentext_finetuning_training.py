import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Le transfer learning consiste à prendre un modèle pré-entraîné sur une grande quantité de données génériques et à le réentraîner
# (fine-tuning) sur un ensemble de données spécifiques à votre domaine. 
# Cela permet au modèle de bénéficier des connaissances générales acquises lors de l'entraînement initial
# tout en s'adaptant aux particularités de votre domaine d'application.

#Étapes pour Fine-tuner un Modèle Préentraîné
#Choisir un Modèle Préentraîné :
#Sélectionnez un modèle de base tel que GPT-2, BERT, RoBERTa, etc., en fonction de vos besoins (génération de texte, classification, Q&A, etc.).
#Préparer vos Données :
#Rassemblez et nettoyez les données spécifiques à votre domaine (par exemple, des documents de lois, des manuels techniques, des articles de recherche).
#Formatez les données pour qu'elles soient compatibles avec le modèle (souvent en fichiers texte ou dans des formats de dataset spécifiques).
#Entraînement du Modèle (Fine-tuning) :
#Utilisez des bibliothèques comme Hugging Face Transformers pour effectuer le fine-tuning.
#Définissez les hyperparamètres de l'entraînement (nombre d'époques, taille du batch, taux d'apprentissage, etc.).
#Évaluer et Ajuster :
#Évaluez les performances du modèle sur un ensemble de données de validation.
#Ajustez les hyperparamètres si nécessaire et réentraînez.
#Déploiement :
#Une fois le modèle fine-tuné et évalué, déployez-le sur votre infrastructure (serveurs locaux, cloud, etc.).
#Intégrez le modèle dans vos applications (par exemple, un chatbot, un système de Q&A).


# Pour des essais gratuits et des projets à petite échelle, DistilGPT-2 et autres modèles open source sont d'excellentes options. 
# Vous pouvez les fine-tuner et les utiliser sans frais supplémentaires.
# Pour des besoins de performance plus élevés et des applications complexes, utiliser GPT-3 via l'API d'OpenAI est une solution
# puissante, bien que payante. Vous aurez besoin d'une clé API et de gérer les coûts associés.


# Chemin vers le fichier texte généré
file_path = "henoch.txt"

# Charger le tokenizer et le modèle pré-entraîné
model_name = 'distilgpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Créer un dataset à partir de votre fichier texte
def load_dataset(file_path, tokenizer, block_size=128):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size
    )

dataset = load_dataset(file_path, tokenizer)

# Créer un DataCollator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Configurer les arguments de l'entraînement
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=32, # Augmenter le nombre d'époques, premier test avec 3 -> mediocre. deuxieme avec 32 ->
    per_device_train_batch_size=5, # Ajuster la taille du batch, premier test avec 3 -> mediocre, deuxieme avec 5
    save_steps=10_000,
    save_total_limit=2,
)

# Créer un Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# Entraîner le modèle
trainer.train()

# Sauvegarder le modèle fine-tuné
model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')