# Proje sadece Main.py veya start.bat dosyalarından çalıştırılabilir.

# main.py
# Hacettepe Üniversitesi Asistanı
# Bu kod, Hacettepe Üniversitesi Eğitim Yönetmeliği belgelerini kullanarak
# kullanıcının sorularını yanıtlamak için bir RAG (Retrieval-Augmented Generation) uygulamasıdır.
# Bu uygulama, belgeleri indeksler ve yanıtlar oluşturur.




import os
import subprocess
from dotenv import load_dotenv
from build_index import build_index

load_dotenv()
json_path = os.getenv("JSON_PATH")
faiss_index_path = os.getenv("FAISS_INDEX_PATH")
embedding_model_name = os.getenv("EMBEDDING_MODEL")
faiss_index_path = os.getenv("FAISS_INDEX_PATH")

# FAISS index var mı?
if not os.path.exists(faiss_index_path):
    print("📦 FAISS index bulunamadı. Oluşturuluyor...")
    # FAISS index oluşturma fonksiyonunu burada çağırın veya tanımlayın
    # Örneğin:
    build_index(faiss_index_path, json_path, embedding_model_name)
else:
    print("✅ FAISS index zaten var.")

# Streamlit app başlat
print("🚀 Uygulama başlatılıyor...")
subprocess.run(["streamlit", "run", "streamlit_app.py"])
