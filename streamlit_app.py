# Proje sadece Main.py veya start.bat dosyalarından çalıştırılabilir.

# streamlit_app.py
# Bu dosya, Hacettepe Üniversitesi Asistanı için bir Streamlit uygulamasıdır.
# Sadece kullanıcı arayüzünü oluşturur ve ana işlevselliği içermez.


import os
import streamlit as st
from llm_pipeline import retriever, rerank_documents, llm, prompt
# from main import embeddings, retriever, rerank_documents, llm, prompt     eski main.py
from langchain.chains import LLMChain
from PIL import Image
import base64
from dotenv import load_dotenv

# Sayfa ayarları
st.set_page_config(page_title="Hacettepe Asistan", layout="centered")

# Kullanılacak görselleri base64 olarak hazırlama
def get_base64_bg(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Hacettepe logosu
load_dotenv()
logo_path = os.getenv("HACETTEPE_LOGO_PATH")
hacettepe_img_base64 = get_base64_bg(logo_path)

# # Arka plan görseli (yapılabilir)
# bg_img_base64 = get_base64_bg("images/hacettepe.png")

# # Geyik logosu (yapılabilir)
# deer_img_base64 = get_base64_bg("images/deer.png")

# st.markdown(
#     """
#     <style>                           # Stramlit header'ını gizle
#     header {visibility: hidden;}
#     .block-container {
#         padding-top: 1rem;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# Arka plan uygula
# st.markdown(
#     f"""
#     <style>
#     .stApp {{
#         background: url("data:image/png;base64,{bg_img_base64}") no-repeat center center fixed;
#         background-size: contain;
#         background-color: black;
    
#     }}
#     header {{ background-color: black; }}
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# Hacettepe logosunu ortala ve göster
st.markdown(
    f"""
    <div style='display: flex; justify-content: center; margin-top: 30px; margin-bottom: 10px;'>
        <img src='data:image/png;base64,{hacettepe_img_base64}' width='120'>
    </div>
    """,
    unsafe_allow_html=True
)

# Başlıklar
st.markdown(
    """
    <style>
    .title {

        font-size: 32px;
        text-align: center;
        font-weight: bold;
    }
    .info-text {
        text-align: center;
        font-size: 16px;
        margin-bottom: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Başlık ve bilgi metni
st.markdown("<div class='title'>Hacettepe Üniversitesi Yapay Zeka Asistanı</div>", unsafe_allow_html=True)
st.markdown("<div class='info-text'>Hacettepe hakkında merak ettiğiniz soruları buradan sorabilirsiniz.</div>", unsafe_allow_html=True)


# Soru için kullanıcı arayüzü
user_question = st.text_input("🔍 Soru:", placeholder="Sorunuzu yazınız...")


# Soru varsa işle
if user_question:
    # Kullanıcıdan gelen soruyu al
    with st.spinner("Kaynak metinler aranıyor..."):
        # Soruya göre belgeleri al
        formatted_query = f"query: {user_question.lower()}"
        retrieved_docs = retriever.invoke(formatted_query)
        top_docs = rerank_documents(formatted_query, retrieved_docs, top_n=10)
        context = "\n\n".join([doc.page_content for doc in top_docs])
    
    with st.spinner(" Asistan yanıt oluşturuyor..."):
        # Prompt zincirini oluştur
        rag_chain = prompt | llm
        response = rag_chain.invoke({
            "question": user_question.lower(),
            "context": context.lower()
        })
    
    # Yanıtı göster
    st.subheader("📤 Cevap (Detaylı bilgi için aşağıdan kaynak metinlere bakınız.)")
    st.markdown(response)

    # Kaynak belgeler
    with st.expander("🔎 Kaynak metinler"):
        for doc in top_docs:
            st.markdown(f"- **{doc.metadata.get('title', 'Belirsiz Başlık')}**\n\n{doc.page_content[:400]}...")