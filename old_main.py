# Bu old_main dosyası aktif olarak çalıştırılmamaktadır. Projenin prototipi için kullanılmıştır.
# Embedding, cross ranking ve llm modeli bu kod içerisinde kontrol edilebilir.

import os
import sys
import json
from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from langchain_community.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Hacettepe Üniversitesi Eğitim Yönetmeliği için RAG uygulaması
# Bu kod, Hacettepe Üniversitesi Eğitim Yönetmeliği belgelerini kullanarak
# kullanıcının sorularını yanıtlamak için bir RAG (Retrieval-Augmented Generation) uygulamasıdır.
# Bu uygulama, belgeleri indeksler ve yanıtlar oluşturur.

# Yollar
load_dotenv()
json_path = os.getenv("JSON_PATH")
faiss_index_path = os.getenv("FAISS_INDEX_PATH")
model_path = os.getenv("MODEL_PATH")
embedding_model_name = os.getenv("EMBEDDING_MODEL")
reranker_model_name = os.getenv("RERANKER_MODEL")

# JSON dosyası var mı kontrol et
try:    
    with open(json_path, "r", encoding="utf-8") as f:
        # JSON dosyasını oku
        raw_data = json.load(f)
    print("✅ JSON dosyası başarıyla yüklendi.")
except FileNotFoundError:
    print(f"❌ HATA: JSON dosyası bulunamadı: {json_path}")
    sys.exit(1)
except json.JSONDecodeError as e:
    print(f"❌ HATA: JSON bozuk veya geçersiz: {e}")
    sys.exit(1)

# # # FAISS dizini varsa sil
# # if os.path.exists(faiss_index_path):
# #     try:
# #         # FAISS dizinini sil
# #         shutil.rmtree(faiss_index_path)
# #         print(f"🗑️ Var olan FAISS index silindi: {faiss_index_path}")
# #     except Exception as e:
# #         # FAISS dizini silinemediğinde hata ver
# #         print(f"❌ HATA: FAISS index silinemedi: {e}")
# #         sys.exit(1)

# HuggingFaceEmbeddings ile Embedding modelini yükle
try:
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    print(f"✅ Embedding modeli yüklendi: {embedding_model_name}")
except Exception as e:
    print(f"❌ HATA: Embedding modeli yüklenemedi: {e}")
    sys.exit(1)

# Text splitter (gerekirse)
splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)

# Belgeleri hazırla ("passage: ..." formatında)
documents = []
# Her bir belge için içerik ve metadata al
for item in raw_data:
    content = item.get("content", "").strip()
    # Eğer içerik yoksa geç
    if not content:
        continue
    # İçeriği küçük harfe çevir ve böl
    chunks = splitter.split_text(content.lower()) 
    # Her bir parça için Document nesnesi oluştur
    for chunk in chunks:
        documents.append(
            Document(
                # "passage: ..." formatında içerik
                page_content=f"passage: {chunk.lower()}",
                metadata={
                    # Metadata bilgileri
                    "article_id": item.get("article_id", ""),   
                    "section": item.get("section", ""), 
                    "title": item.get("title", "")
                }
            )
        )

# FAISS index önceden varsa yeniden oluşturma
if os.path.exists(faiss_index_path):
    print(f"ℹ️ FAISS index zaten var, yeniden oluşturulmadı: {faiss_index_path}")
    try:
        # FAISS dizinini yükle ve retriever olarak ayarla
        db = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)   
        retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 20})
        print("✅ FAISS index yüklendi ve retriever hazır.")
    except Exception as e:
        print(f"❌ HATA: Mevcut FAISS index yüklenemedi: {e}")
        sys.exit(1)
else:

    print("📦 FAISS index bulunamadı, yeniden oluşturuluyor...")
    try:
        # FAISS dizinini oluştur, kaydet ve retriever olarak ayarla
        db = FAISS.from_documents(documents, embeddings)
        db.save_local(faiss_index_path)
        retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 20})
        print(f"✅ FAISS index başarıyla oluşturuldu, kaydedildi ve retriever hazır.")
    except Exception as e:
        print(f"❌ HATA: FAISS index oluşturulamadı: {e}")
        sys.exit(1)

# Reranker modelini yükle
try:
    reranker = CrossEncoder(reranker_model_name, device="cpu")
    print(f"✅ Reranker modeli yüklendi: {reranker_model_name}")
except Exception as e:
    print(f"❌ HATA: Reranker modeli yüklenemedi: {e}")
    sys.exit(1)

# Rerank fonksiyonu
def rerank_documents(query, documents, top_n=10):
    # Rerank için belgeleri ve sorguyu uygun formata getir
    pairs = [(query.lower(), doc.page_content.lower()) for doc in documents]
    scores = reranker.predict(pairs)
    # Belgeleri skorlara göre sırala
    reranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in reranked[:top_n]]

# # Örnek Soru Al ve Rerank Et (streamlit_app.py de)
# query = input("\n📝 Soru (çıkmak için q): ").strip().lower()
# if query.lower() in ["q", "quit", "exit"]:
#     sys.exit(0)

# Formatı embedding modeline uygun hale getir (streamlit_app.py de)
# formatted_query = f"query: {query.lower()}"  

# # FAISS'ten istediğimiz sayıda belgeyi getir ve kontrol et   (streamlit_app.py de)
# try:
#     retrieved_docs = retriever.invoke(formatted_query)
#     # retrieved_docs = retriever.get_relevant_documents(formatted_query)
#     if not retrieved_docs:
#         print("❌ HATA: FAISS'ten belge alınamadı (Boş sonuç döndü).")
#         sys.exit(1)
# except Exception as e:
#     print(f"❌ HATA: FAISS sorgusu sırasında hata oluştu: {e}")
#     sys.exit(1)

# # Kontrol için FAISS output belgelerini yazdır
# print(f"\n🔎 FAISS ilk {len(retrieved_docs)} belge getirdi.")
# print("\n📘 E5-Large FAISS 20 belge :\n")
# for i, doc in enumerate(retrieved_docs, 1):
#     print(f"{i}. {doc.metadata.get('article_id', '')} - {doc.metadata.get('title', '')}")
#     print(doc.page_content[:350] + "...\n")

# Rerank ile en iyi 10 sırala (streamlit_app.py de)
# top_docs = rerank_documents(formatted_query, retrieved_docs, top_n=10)

# # En iyi belgeleri yazdır
# print("\n📘 En iyi 10 belge (RERANKED):\n")
# for i, doc in enumerate(top_docs, 1):
#     # Her bir belgenin başlığını ve içeriğini yazdır
#     print(f"{i}. {doc.metadata.get('article_id', '')} - {doc.metadata.get('title', '')}")
#     print(doc.page_content[:250] + "...\n")

# Gemma modeli (LlamaCpp) ayarları
llm = LlamaCpp(
    model_path=model_path,  
    temperature=0.1,    
    max_tokens=512, 
    n_ctx=8192, 
    n_threads=8,
    n_gpu_layers=20,
    top_p=0.9,
    verbose=False
)

# Prompt şablonu
prompt_template = """Sen Hacettepe Üniversitesi için uzmanlaşmış bir asistansın. 
Tek görevin, Hacettepe Üniversitesi Ön Lisans, Lisans Eğitim–Öğretim Yönetmeliği hakkında SAĞLANAN BAĞLAMA dayanarak soruları yanıtlamaktır.
Soruyu yanıtlamak için aşağıdaki alınmış bağlam parçalarını kullan.
Eğer soru Hacettepe Üniversitesi yönetmeliği ile ilgili değilse (örneğin spor, politika, siyaset, ekonomi, genel bilgi, yönetmelik bağlamı dışındaki selamlamalar),
bu konuyla yardımcı olamayacağını ve yalnızca sağlanan belgelere dayanarak Hacettepe Üniversitesi yönetmeliği hakkında bilgi verebileceğini belirt.
Eğer sağlanan bağlam yönetmelikle ilgili bir sorunun cevabını içermiyorsa, bilginin sağlanan belgelerde bulunmadığını belirt.
Kullanıcı başka bir dilde sormadığı sürece Türkçe yanıt ver. Kesin ol ve bağlamdaki bilgilere sadık kal. Detaylı ve açıklayıcı ol.
Soru: {question}
=========
{context}
=========
Cevap:"""

# # Prompt şablonu (Openchat ile uyumlu)
# prompt_template = """<|system|>propmt<|end|>
# <|user|>
# Soru: {question}
# Bağlam:{context}
# <|end|>
# <|assistant|>""" 

# # prompt tanımı   (alternatif)
# prompt = PromptTemplate.from_template("Soru: {question}\nCevap:")

# PromptTemplate ile şablonu oluştur
prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["question", "context"]
)

# # LLMChain ile zinciri oluştur (alternatif)
# chain = LLMChain(llm=llm, prompt=prompt)

# LLMChain ile zinciri oluştur
rag_chain = prompt | llm

# Seçilen top 10 dokümandan context hazırla (streamlit_app.py de)
# context = "\n\n".join([doc.page_content for doc in top_docs])

# LLM'e gönder ve cevap al (alternatif)
# response = chain.invoke({
#     "question": query,
#     "context": context
# })

# LLM'e gönder ve cevap al (streamlit_app.py de)
# response = rag_chain.invoke({
#     "question": query.lower(),
#     "context": context.lower()
# })

# # Cevabı yazdır (streamlit_app.py de)
# print("\n🧠 DeerBot'dan Cevap:\n")
# print(response)