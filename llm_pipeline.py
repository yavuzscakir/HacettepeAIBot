# Proje sadece Main.py veya start.bat dosyalarından çalıştırılabilir.

# Projenin llm pipeline'ını oluşturan dosyası

import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Ortam değişkenlerini yükle
load_dotenv()

# Ayarlar
faiss_index_path = os.getenv("FAISS_INDEX_PATH")
embedding_model_name = os.getenv("EMBEDDING_MODEL")
reranker_model_name = os.getenv("RERANKER_MODEL")
model_path = os.getenv("MODEL_PATH")

# Embedding modelini yükle
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

# FAISS index'i yükle
db = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 20})

# Reranker modelini yükle
reranker = CrossEncoder(reranker_model_name, device="cpu")

# Rerank fonksiyonu
def rerank_documents(query, documents, top_n=10):
    pairs = [(query.lower(), doc.page_content.lower()) for doc in documents]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in reranked[:top_n]]

# LlamaCpp modeli (Gemma)
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

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["question", "context"]
)
