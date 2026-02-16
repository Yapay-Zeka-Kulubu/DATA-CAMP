import os
import re
from google import genai

client = genai.Client(api_key="apiyi bruaya ekle")
MODEL_ID = "gemini-2.5-flash" 

current_dir = os.path.dirname(os.path.abspath(__file__)) #dosya yolu oku
file_path = os.path.join(current_dir, "data", "document.txt")

def load_documents(path):
    if not os.path.exists(path):
        print(f"HATA: Dosya bulunamadı: {path}")
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

document_text = load_documents(file_path)

if not document_text:
    print("UYARI: document.txt bulunamadı veya boş.") #boşsa program hata almasın
    paragraphs = []
else:
    paragraphs = [p.strip() for p in document_text.splitlines() if p.strip()]

#arama fonks.
def extract_keywords(question: str):
    question = question.lower()
    words = re.findall(r"\b\w+\b", question)
    return set(words)

def retrieve_relevant_paragraphs(paragraphs, question, top_k=3):
    keywords = extract_keywords(question)
    scored = []

    for p in paragraphs:
        p_words = set(re.findall(r"\b\w+\b", p.lower()))
        score = len(keywords & p_words)
        if score > 0:
            scored.append((score, p))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [p for _, p in scored[:top_k]]


user_query = input("Sorunuzu giriniz:")

relevant_chunks = []
if paragraphs:
    relevant_chunks = retrieve_relevant_paragraphs(paragraphs, user_query)

if not relevant_chunks:
    print("\nCevap: Bu bilgi dökümanda bulunmamaktadır.")
else:
    context = "\n\n".join(relevant_chunks)
    
    prompt = f"SORU: {user_query}\n\nBAĞLAM:\n{context}"
    
    system_instruction = (
        "Sen sadece verilen bağlama göre cevap veren bir asistansın. "
        "Bağlam dışına asla çıkma. Eğer cevap bağlamda yoksa "
        "'Bu bilgi dökümanda bulunmamaktadır' de."
    )

    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config={
                "system_instruction": system_instruction
            }
        )
        print("\nCevap:")
        print(response.text)
    except Exception as e:
        print(f"\nAPI Hatası oluştu: {e}")