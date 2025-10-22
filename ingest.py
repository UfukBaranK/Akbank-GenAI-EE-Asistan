import os
from dotenv import load_dotenv
# DirectoryLoader'ı ekle. Bu, klasördeki tüm PDF'leri otomatik bulur.
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
    
load_dotenv()

# KRİTİK BÖLÜM: API Anahtarını Kütüphanenin Okuyacağı Şekilde Garanti Altına Al
API_KEY_VALUE = os.getenv("GEMINI_API_KEY")
if not API_KEY_VALUE:
    API_KEY_VALUE = os.getenv("GOOGLE_API_KEY") 
    
if API_KEY_VALUE:
    os.environ["GEMINI_API_KEY"] = API_KEY_VALUE
    os.environ["GOOGLE_API_KEY"] = API_KEY_VALUE # SDK'nın okuması için set et
else:
    print("🚨 HATA: .env dosyasında geçerli bir API Anahtarı bulunamadı.")
    
EMBEDDING_MODEL = "models/embedding-001" 
VECTOR_DB_PATH = "./vector_db"
PDF_DIR = "data" # PDF'lerin bu klasörde olmalı

def create_vector_db():
    print("💡 EE Ders Kitapları Yükleniyor (DirectoryLoader ile toplu yükleme)...")

    # API Key kontrolü
    if not os.environ.get("GEMINI_API_KEY"):
        return
    
    # Klasör kontrolü
    if not os.path.exists(PDF_DIR):
        print(f"🚨 HATA: Belirtilen '{PDF_DIR}' klasörü bulunamadı!")
        return
        
    # YENİ YÜKLEME MANTIĞI: DirectoryLoader kullanarak klasördeki tüm PDF'leri bul
    try:
        loader = DirectoryLoader(
            path=PDF_DIR,
            glob="**/*.pdf", # Sadece PDF dosyalarını arar
            loader_cls=PyPDFLoader, # PyPDFLoader'ı her dosya için kullanır
            show_progress=True, # Yüklemeyi izle
            use_multithreading=True # Çoklu çekirdek kullanarak daha hızlı yükle
        )
        documents = loader.load()
    except Exception as e:
        print(f"🚨 HATA: Dosya yüklenirken bir sorun oluştu: {e}")
        return
        
    if not documents:
        print(f"🚨 HATA: '{PDF_DIR}' klasöründe yüklenecek PDF bulunamadı.")
        return

    # Metin bölme ve Vektör Veritabanı oluşturma
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    print(f"Toplam {len(documents)} sayfa yüklendi, {len(texts)} bilgi parçasına bölündü.")

    # API Anahtarını manüel geçirmeyi kaldırdık. Kütüphane ortam değişkeninden okuyacak.
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL) 
    
    # ChromaDB'ye yükle (Var olanı silip yenisini yazar)
    Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=VECTOR_DB_PATH
    )
    print(f"✅ Vektör veritabanı başarıyla oluşturuldu: '{VECTOR_DB_PATH}'")

if __name__ == "__main__":
    create_vector_db()

# 🏃 Çalıştırma Komutu: python ingest.py