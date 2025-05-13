# Proje sadece Main.py veya start.bat dosyalarından çalıştırılabilir.

# build_index.py    
# Hacettepe Üniversitesi Asistanı için FAISS index oluşturma
# Bu kod, Hacettepe Üniversitesi Eğitim Yönetmeliği belgelerini kullanarak
# belgeleri parçalayıp FAISS index oluşturur.


import os
import json
import sys
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Ortam değişkenlerini yükle
load_dotenv()

# Gerekli yollar ve model adı

def build_index(faiss_index_path, json_path, embedding_model_name):
    # JSON dosyasını oku
    try:    
        with open(json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        print("✅ JSON dosyası başarıyla yüklendi.")
    except FileNotFoundError:
        print(f"❌ HATA: JSON dosyası bulunamadı: {json_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ HATA: JSON bozuk veya geçersiz: {e}")
        sys.exit(1)

    # Embedding modelini yükle
    try:
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        print(f"✅ Embedding modeli yüklendi: {embedding_model_name}")
    except Exception as e:
        print(f"❌ HATA: Embedding modeli yüklenemedi: {e}")
        sys.exit(1)

    # Text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    # Belgeleri oluştur
    documents = []
    for item in raw_data:
        if isinstance(item, dict):
            for key, value in item.items():
                if isinstance(value, str):
                    documents.append(Document(page_content=value, metadata={"source": key}))
                elif isinstance(value, list):
                    for sub_item in value:
                        if isinstance(sub_item, str):
                            documents.append(Document(page_content=sub_item, metadata={"source": key}))

    # Belgeleri böl
    docs = text_splitter.split_documents(documents)

    # FAISS index'i oluştur
    db = FAISS.from_documents(docs, embeddings)
    
    # FAISS index'i kaydet
    db.save_local(faiss_index_path)
    
    print(f"✅ FAISS index başarıyla oluşturuldu ve kaydedildi: {faiss_index_path}")
    # JSON dosyasını yükle
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        print("✅ JSON dosyası yüklendi.")
    except Exception as e:
        print(f"❌ JSON dosyası yüklenemedi: {e}")
        exit(1)


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

# Embedding modeli yükle
    try:
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        print(f"✅ Embedding modeli yüklendi: {embedding_model_name}")
    except Exception as e:
        print(f"❌ Embedding yükleme hatası: {e}")
        exit(1)

    # Text splitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)

    # Belgeleri parçalayıp hazırla
    documents = []
    for item in raw_data:
        content = item.get("content", "").strip()
        if not content:
            continue
        chunks = splitter.split_text(content.lower())
        for chunk in chunks:
            documents.append(Document(
                page_content=f"passage: {chunk}",
                metadata={
                    "article_id": item.get("article_id", ""),
                    "section": item.get("section", ""),
                    "title": item.get("title", "")
                }
            ))

    print(f"🔧 Toplam {len(documents)} belge oluşturuldu.")

    # FAISS index oluştur
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