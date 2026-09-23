import streamlit as st
import time
from enum import Enum
from typing import List, Optional, Dict, Any
from groq import Groq
import os
import datetime
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter



# ====================================================
#                 ENUMS & CONSTANTS
# ====================================================
class MessageType(Enum):
    USER = "user"
    ASSISTANT = "assistant"

# ====================================================
#                 GROQ CLIENT CLASS
# ====================================================
class GroqClient:
    def __init__(self, api_key: str = "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", model: str = "meta-llama/llama-4-maverick-17b-128e-instruct"):
        self.client = Groq(api_key=api_key)
        self.model = model
    
    def generate_response(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=2048,
                top_p=1,
                stream=False
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"Hata oluştu: {str(e)}"

# ====================================================
#                 CHAT MANAGER CLASS (GÜNCELLENDİ)
# ====================================================
class ChatManager:
    def __init__(self):        
        # Tüm sohbetleri tutan ana liste
        if "all_chats" not in st.session_state:
            st.session_state.all_chats = []
        
        # Şu anki aktif sohbetin ID'si
        if "current_chat_id" not in st.session_state:
            self.create_new_chat()
        
        # RAG için dosya metni (ham text)
        if "file_content" not in st.session_state:
            st.session_state.file_content = None

    def create_new_chat(self):
        """Yeni bir boş sohbet oluşturur ve aktif yapar."""
        new_id = len(st.session_state.all_chats)
        new_chat = {
            "id": new_id,
            "title": "Yeni Sohbet",
            "messages": [],
            "timestamp": datetime.datetime.now()
        }
        st.session_state.all_chats.append(new_chat)
        st.session_state.current_chat_id = new_id
        return new_id

    def get_current_chat(self):
        """Aktif sohbet objesini döndürür."""
        chat_id = st.session_state.current_chat_id
        for chat in st.session_state.all_chats:
            if chat["id"] == chat_id:
                return chat
        return None

    def add_message(self, role: str, content: str):
        """Aktif sohbete mesaj ekler."""
        current_chat = self.get_current_chat()
        if current_chat:
            current_chat["messages"].append({"role": role, "content": content})
            
            if len(current_chat["messages"]) == 1 and role == "user":
                title = content[:30] + "..." if len(content) > 30 else content
                current_chat["title"] = title

    def switch_chat(self, chat_id):
        """Başka bir sohbete geçiş yapar."""
        st.session_state.current_chat_id = chat_id
    
    def load_file_content(self, file):
        """Yüklenen dosyayı okuyup metin döndürür."""
        if file.name.endswith(".pdf"):
            temp_path = "temp.pdf"
            with open(temp_path, "wb") as f:
                f.write(file.read())
            loader = PyPDFLoader(temp_path)
            docs = loader.load()
            return "\n".join([d.page_content for d in docs])

        elif file.name.endswith(".txt"):
            return file.read().decode("utf-8")

        elif file.name.endswith(".docx"):
            temp_path = "temp.docx"
            with open(temp_path, "wb") as f:
                f.write(file.read())
            loader = Docx2txtLoader(temp_path)
            docs = loader.load()
            return "\n".join([d.page_content for d in docs])

        else:
            return "Desteklenmeyen dosya formatı."
    
    def find_relevant_context(self, user_message: str, file_content: str, max_chars: int = 2000) -> str:
        """Kullanıcı sorusuna göre dosyadan en ilgili bölümü bulur."""
        if not file_content:
            return ""
        
        # Dosyayı paragraflara böl
        paragraphs = file_content.split('\n\n')
        
        # Kullanıcı sorusundaki anahtar kelimeleri bul
        keywords = user_message.lower().split()
        
        # Her paragrafın ilgililik skorunu hesapla
        scored_paragraphs = []
        for para in paragraphs:
            if len(para.strip()) < 10:  # Çok kısa paragrafları atla
                continue
            
            para_lower = para.lower()
            score = sum(1 for keyword in keywords if keyword in para_lower)
            
            if score > 0:
                scored_paragraphs.append((score, para))
        
        # En yüksek skorlu paragrafları al
        scored_paragraphs.sort(reverse=True, key=lambda x: x[0])
        
        # İlgili bağlamı oluştur
        context = ""
        for score, para in scored_paragraphs[:5]:  # En iyi 5 paragraf
            if len(context) + len(para) < max_chars:
                context += para + "\n\n"
            else:
                break
        
        # Eğer hiç eşleşme yoksa, dosyanın başından bir kısmını al
        if not context:
            context = file_content[:max_chars]
        
        return context.strip()
        
    def generate_response(self, user_message: str):
        api_key = "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"

        # --- AKILLI RAG SİSTEMİ ---
        system_prompt = "Sen yardımsever bir AI asistansın. Türkçe cevap ver."
        
        # Eğer dosya yüklenmişse, akıllı bağlam bulma
        if st.session_state.file_content:
            # Vector DB yerine akıllı metin arama kullan
            relevant_context = self.find_relevant_context(
                user_message, 
                st.session_state.file_content,
                max_chars=2000  # Maksimum 2000 karakter bağlam
            )
            
            if relevant_context:
                system_prompt += (
                    f"\n\nAşağıdaki dosya içeriğine dayanarak cevap ver:\n\n"
                    f"{relevant_context}\n\n"
                    f"Cevabın dosyada verilen bilgilere uygun ve detaylı olsun. "
                    f"Eğer kullanıcı dosya hakkında genel bir soru soruyorsa, dosyanın içeriğini özetleyerek açıkla."
                )

        messages_for_api = [
            {"role": "system", "content": system_prompt}
        ]

        current_chat = self.get_current_chat()
        if current_chat:
            # Son 3 mesajı al ve her mesajı 400 karakterle sınırla
            recent_messages = current_chat["messages"][-3:]
            for msg in recent_messages:
                messages_for_api.append({
                    "role": msg["role"],
                    "content": msg["content"][:400]
                })

        client = GroqClient(api_key=api_key)
        return client.generate_response(messages_for_api)

# ====================================================
#                 STYLING
# ====================================================
class StyleManager:
    @staticmethod
    def apply_styles():
        st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        /* Genel Ayarlar */
        h1 {
            font-family: 'Inter', sans-serif;
            font-size: 2rem !important;
            font-weight: 700 !important;
            color: #ffffff !important;
            text-align: center;
            margin-bottom: 1rem !important;
        }

        /* Sidebar Logosu için ayar */
        [data-testid="stSidebar"] img {
            border-radius: 15px;
            margin-bottom: 20px;
           
        }

        /* Hoşgeldin ekranı */
        .welcome-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            margin-top: 1rem;
            margin-bottom: 2rem;
            text-align: center;
        }
        
        /* Butonlar */
        .stButton > button {
            width: 100%;
            border-radius: 8px;
            border: 1px solid #4a4a4a;
        }
        </style>
        """, unsafe_allow_html=True)

# ====================================================
#                 MAIN VIEW
# ====================================================
class MainView:
    @staticmethod
    def render_welcome():
        logo_path = "/Users/w/Desktop/Kodlama/VsCode/HelloWorld/erciyesyapayzeka/ClubChatBot/frontend/assets/fav1.png"
        
        col_left, col_center, col_right = st.columns([1, 0.6, 1])
        with col_center:
            if os.path.exists(logo_path):
                st.image(logo_path, width=180) 
            else:
                st.image("https://cdn-icons-png.flaticon.com/512/4712/4712027.png", width=150)
        
        st.markdown("""
            <div class="welcome-container">
                <h1>Keşfedilmiş Kainatın En İyi Kulübüne Hoş Geldiniz 🚀</h1>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        selection = None
        
        with col1:
            if st.button("🐍 Python'da liste nasıl oluşturulur?", use_container_width=True): 
                selection = "Python'da liste nasıl oluşturulur?"
            if st.button("✍️ Bana yaratıcı bir hikaye anlat", use_container_width=True): 
                selection = "Bana yaratıcı bir hikaye anlat"

        with col2:
            if st.button("🔌 API entegrasyonu nasıl yapılır?", use_container_width=True): 
                selection = "API entegrasyonu nasıl yapılır?"
            if st.button("📊 Veri analizi araçları nelerdir?", use_container_width=True): 
                selection = "Veri analizi için en iyi araçlar nelerdir?"
        
        return selection

# ====================================================
#                 MAIN APP FLOW
# ====================================================
def main():
    st.set_page_config(page_title="Yapay Zeka Kulübü", page_icon="🤖", layout="centered")
    StyleManager.apply_styles()
    chat_manager = ChatManager()
    
    uploaded_file = st.file_uploader("📄 Bir dosya yükleyin (PDF / TXT / DOCX)", type=["pdf", "txt", "docx"])

    # Dosya yüklendiğinde içeriği kaydet (Vector DB yerine ham metin)
    if uploaded_file:
        if "last_uploaded_file" not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
            st.info("Dosya işleniyor...")
            
            file_text = chat_manager.load_file_content(uploaded_file)
            st.session_state.file_content = file_text
            st.session_state.last_uploaded_file = uploaded_file.name
            
            st.success(f"✅ Dosya yüklendi! ({len(file_text)} karakter)")
            st.success("Artık bu dosya hakkında soru sorabilirsiniz.")
    
    # --- AYARLAR ---
    ai_avatar_path = "/Users/w/Desktop/Kodlama/VsCode/HelloWorld/erciyesyapayzeka/ClubChatBot/frontend/assets/fav1.png"
    sidebar_logo_path = "/Users/w/Desktop/Kodlama/VsCode/HelloWorld/erciyesyapayzeka/ClubChatBot/frontend/assets/logo.png"

    # ================= SIDEBAR =================
    with st.sidebar:
        if os.path.exists(sidebar_logo_path):
            st.image(sidebar_logo_path, use_container_width=True)
        else:
            st.warning("Sidebar logosu bulunamadı.")

        st.title("Sohbetler")
        
        if st.button("➕ Yeni Sohbet Başlat", type="primary", use_container_width=True):
            chat_manager.create_new_chat()
            st.rerun()
            
        st.markdown("---")
        st.caption("GEÇMİŞ SOHBETLER")

        for chat in reversed(st.session_state.all_chats):
            if st.button(f"💬 {chat['title']}", key=f"chat_btn_{chat['id']}", use_container_width=True):
                chat_manager.switch_chat(chat['id'])
                st.rerun()

    # ================= ANA İÇERİK =================
    current_chat = chat_manager.get_current_chat()
    
    # --- 1. GEÇMİŞ MESAJLARI GÖSTER ---
    if not current_chat["messages"]:
        selected_prompt = MainView.render_welcome()
        if selected_prompt:
            chat_manager.add_message("user", selected_prompt)
            st.rerun()
    else:
        for msg in current_chat["messages"]:
            if msg["role"] == "assistant":
                current_avatar = ai_avatar_path
            else:
                current_avatar = "👤"
            
            with st.chat_message(msg["role"], avatar=current_avatar):
                st.markdown(msg["content"])

    # --- 2. YENİ MESAJ VE CEVAP ---
    if prompt := st.chat_input("Mesajınızı buraya yazın..."):
        # Kullanıcı Mesajı
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        chat_manager.add_message("user", prompt)

        # Asistan Cevabı
        with st.chat_message("assistant", avatar=ai_avatar_path):
            with st.spinner("Düşünüyorum..."):
                response = chat_manager.generate_response(prompt)
                st.markdown(response)
        chat_manager.add_message("assistant", response)
        
        st.rerun()

if __name__ == "__main__":
    main()