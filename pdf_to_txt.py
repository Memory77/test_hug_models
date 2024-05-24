import fitz  # PyMuPDF

# Chemin vers le fichier PDF
pdf_path = "LE-LIVRE-D-HENOCH-COMPLET-1.pdf"
text_output_path = "henoch.txt"

# Ouvrir le PDF
pdf_document = fitz.open(pdf_path)

# Extraire le texte de chaque page et l'écrire dans un fichier texte
with open(text_output_path, "w", encoding="utf-8") as text_file:
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text = page.get_text()
        text_file.write(text)

print(f"Le texte a été extrait du PDF et sauvegardé dans {text_output_path}")