# HacettepeAIBot
Hacettepe University Artificial Intelligence Assistant Bot

🧠 HacettepeAIBot

Hacettepe Üniversitesi Ön Lisans ve Lisans Yönetmeliği üzerine uzmanlaşmış bir Yapay Zekâ Yardımcı Botu. Kullanıcıdan gelen soruları işleyerek, yönetmeliğe dayalı cevaplar üretir.

🚀 Nasıl Çalışır?

Projeyi başlatmak için terminalde:

start.bat

Bu betik, sistemde gerekli kontrolleri yapar ve ardından main.py üzerinden uygulamayı başlatır.

.
├── assets/                         # Görsel, ikon, stil dosyaları vb.

├── data/                           # Kaynak veri, doküman veya yedekler

├── faiss_indexes/yonetmelik_index/ # FAISS vektör indeks klasörü (otomatik oluşur)

├── models                          # gemma-2b-it.Q4_K_M.gguf (1.6gb) llm modeli

│

├── build_index.py                 # FAISS index oluşturma fonksiyonu

├── main.py                        # Ana kontrol akışı (index kontrolü ve başlatma)

├── llm_pipeline.py                # LLM, retriever, reranker pipeline

├── streamlit_app.py              # Kullanıcı arayüzü (Streamlit üzerinden)

├── start.bat                     # Uygulama başlatıcı Windows betiği

│

├── requirements.txt              # Gerekli temel Python kütüphaneleri

├── full_requirements.txt         # Genişletilmiş bağımlılık listesi

├── old_main.py                   # Yedeklenmiş/önceki sürüm main dosyası

├── README.md                     # Proje açıklaması ve kurulum rehberi


🔧 Ana Bileşenler

✅ main.py

Projenin giriş noktasıdır.

FAISS index yoksa otomatik olarak build_index() fonksiyonunu çalıştırır.

Modeli ve embedding’leri yükleyip chatbotu hazırlar.


✅ build_index.py
yonetmelik_v5.json datasetinden belge okur.

Chunk'lara böler ve HuggingFaceEmbeddings ile vektörleştirir.

FAISS kullanarak vektör indeks oluşturur ve faiss_indexes/ altında kaydeder.


✅ llm_pipeline.py

Embedding retriever, cross-encoder reranker ve LLM (Gemma 2B, Openchat) birleşimini tanımlar.

Asıl soru-cevap zinciri burada tanımlanır.


✅ streamlit_app.py

Web arayüzü sunar.
Kullanıcıdan soru alır, modelin cevabını gösterir.

Kaynak metinleri de kullanıcıya sunar.


✅ start.bat

Otomatik başlatma betiğidir.

Python ortamını ve uygulamayı başlatmak için uygundur.


⚙️ Kurulum

1. Ortamı oluştur:
   
python -m venv venv

venv\Scripts\activate


2. Gerekli kütüphaneleri yükle:
   
pip install -r requirements.txt


3. .env dosyasını oluştur:

4. Uygulamayı başlat:

start.bat      #main.py dosyasından da başlatılabilir.


📦 Kullanılan Teknolojiler

🧠 LangChain

🔍 FAISS

🔤 HuggingFace Embeddings

💬 LLM (Gemma 2B, Openchat, vb.)

🌐 Streamlit

📄 .env + dotenv


📘 Amaç

Bu botun amacı, Hacettepe Üniversitesi öğrencilerinin yönetmeliklerle ilgili sorularını hızlı ve doğru şekilde cevaplamaktır. RAG (Retrieval-Augmented Generation) mimarisi sayesinde, yanıtlar yalnızca resmi belge içeriğine dayalıdır.
