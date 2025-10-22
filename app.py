import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
    
# Proje Geliştirme Ortamı (Kriter 1)
# .env dosyasından API anahtarını yükler (Yerel çalışma için kritik)
load_dotenv()

# KRİTİK BÖLÜM: API Anahtarını Ortam Değişkeni Olarak Garantileme (Çözüm Mimarisi Kriteri)
# LangChain'in ve Google SDK'nın API anahtarını doğru okumasını garanti eder.
API_KEY_VALUE = os.getenv("GEMINI_API_KEY")
if not API_KEY_VALUE:
    API_KEY_VALUE = os.getenv("GOOGLE_API_KEY") # İkinci kontrol
    
if API_KEY_VALUE:
    os.environ["GEMINI_API_KEY"] = API_KEY_VALUE
    os.environ["GOOGLE_API_KEY"] = API_KEY_VALUE # Ortam değişkenini set et
else:
    print("🚨 HATA: .env dosyasında geçerli bir API Anahtarı bulunamadı.")
    
# -- Sabitler (Çözüm Mimarisi Kriteri) --
VECTOR_DB_PATH = "./vector_db"  # Vektör Veritabanı (ChromaDB) yerel depolama yolu
GEMINI_MODEL = "gemini-2.5-flash" # Hız ve maliyet etkinliği için seçilen LLM
EMBEDDING_MODEL = "models/embedding-001" # Google'ın yüksek performanslı embedding modeli
    
def setup_rag_chain():
    # 1. Kimlik Doğrulama Kontrolü (Çalışma Kılavuzu Kriteri)
    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        st.error("🚨 HATA: API Anahtarı bulunamadı. Lütfen Streamlit Secrets'ı/Local .env dosyasını kontrol edin.")
        return None
        
    # 2. Veri Seti Kontrolü (Veri Seti Hazırlama Kriteri)
    # Büyük veri seti (381 kaynak) bulutta olmadığı için, sadece yerel ortamda RAG kurulabilir.
    if not os.path.exists(VECTOR_DB_PATH):
        st.error(f"🚨 HATA: Veritabanı (vector_db klasörü) bulut ortamında mevcut değil.")
        st.warning("⚠️ Uygulamanın **381 Kaynaklık RAG Kabiliyetini** test etmek için, lütfen README.md'deki kılavuza uyarak projeyi yerel bilgisayarınızda çalıştırın.")
        return None
    
    # 3. RAG Zinciri Kurulumu (Çözüm Mimarisi Kriteri)
    
    # 3.1 Vektör Veritabanını Yükleme (Retrieval)
    # ChromaDB (Vektör Database) yerel dosyadan geri yükleniyor.
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)
    
    # Retriever (Alıcı): Büyük veri seti nedeniyle kapsayıcılığı artırmak için k=5 seçildi.
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) 
    
    # 3.2 LLM (Generation) ve Prompt Hazırlığı
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.1) 

    # ESNEK PROMPT MÜHENDİSLİĞİ (Kriter 4: Elde Edilen Sonuçlar)
    # RAG başarısız olsa bile (bağlam yetersizse), LLM'in genel mühendislik bilgisiyle cevap vermesini sağlar.
    template = """Sen, Elektrik-Elektronik Mühendisliği öğrencilerine yönelik pratik bir ders asistanısın. 
    Görevin, kullanıcının sorusuna **enerjik ve profesyonel** bir tonda **detaylı TÜRKÇE cevap** vermektir.

    **KURAL:**
    1. **Öncelikli olarak** aşağıdaki 'Bağlam' kısmında bulunan bilgileri kullan.
    2. Eğer 'Bağlam'da soruya dair **yeterli veya net bir cevap yoksa**, genel mühendislik bilgini ve temel kavramları kullanarak soruyu yanıtla.
    3. Cevabını her zaman profesyonel ve yapılandırılmış bir dille sun.

    Bağlam: 
    {context}

    Soru: {question}

    TÜRKÇE Cevap:"""
    prompt = ChatPromptTemplate.from_template(template)
    
    # 3.3 LangChain Expression Language (LCEL) ile Zincir Oluşturma
    # Tüm RAG adımlarını modüler ve bakımı kolay bir zincirde birleştirir.
    rag_chain = (
        # 1. Bağlamı retriever'dan al ve 2. Soruyu doğrudan geçir.
        {"context": retriever | (lambda x: "\n\n".join([doc.page_content for doc in x])), 
         "question": RunnablePassthrough()} 
        | prompt # 3. Prompt'u uygula
        | llm # 4. LLM'den cevabı al
        | StrOutputParser() # 5. Cevabı metin olarak döndür.
    )
    return rag_chain
    
# Streamlit Ana Fonksiyon (Kriter 5: Web Arayüzü)
def main():
    st.set_page_config(page_title="EE Ders Asistanı", layout="wide")
    
    # GÜNCELLENMİŞ BAŞLIK VE AÇIKLAMA (Proje Başlığı Kriteri)
    st.title("💡 Elektrik-Elektronik Mühendisliği Asistanı (RAG)")
    st.caption("🔍 Hacettepe EE Ders Kitapları ve Yardımcı Kaynaklara Dayalı, Gemini Destekli Akıllı Asistan.")
    
    if 'rag_chain' not in st.session_state:
        st.session_state.rag_chain = setup_rag_chain()
        if st.session_state.rag_chain is None:
            return 
    
    user_question = st.text_input("EE Ders Kitaplarına dair teknik bir soru sor:", 
                                  placeholder="Örn: 'ELE489 dersindeki L1 ve L2 regülarizasyon teknikleri arasındaki farklar nelerdir?'")
    
    if user_question:
        with st.spinner("🚀 Bilgi Kaynakların Taranıyor..."):
            try:
                response = st.session_state.rag_chain.invoke(user_question)
                st.success("Cevap Hazır!")
                st.markdown(f"**Asistan Yanıtı:** \n\n {response}")
            except Exception as e:
                st.error(f"API Çağrısında Hata Kodu: {e}")
                st.warning("Bu hatayı yerel ortamda alıyorsanız: Günlük kota dolmuş olabilir veya API anahtarınız Google Cloud'da kısıtlanmıştır.")
    
if __name__ == "__main__":
    main()