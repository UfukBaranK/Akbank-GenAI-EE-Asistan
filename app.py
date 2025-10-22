import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
    
# .env dosyasındaki anahtarları yükle
load_dotenv()

# KRİTİK BÖLÜM: API Anahtarını Kütüphanenin Okuyacağı Şekilde Garanti Altına Al
API_KEY_VALUE = os.getenv("GEMINI_API_KEY")
if not API_KEY_VALUE:
    API_KEY_VALUE = os.getenv("GOOGLE_API_KEY") # GOOGLE_API_KEY'i de kontrol et
    
if API_KEY_VALUE:
    os.environ["GEMINI_API_KEY"] = API_KEY_VALUE
    os.environ["GOOGLE_API_KEY"] = API_KEY_VALUE # SDK'nın okuması için set et
else:
    print("🚨 HATA: .env dosyasında geçerli bir API Anahtarı bulunamadı.")
    
# -- Sabitler --
VECTOR_DB_PATH = "./vector_db"
GEMINI_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "models/embedding-001"
    
def setup_rag_chain():
    # API Anahtarının varlığını kontrol et
    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        st.error("🚨 HATA: API Anahtarı bulunamadı. Lütfen .env dosyasını kontrol edin.")
        return None
        
    # Veritabanı kontrolü
    if not os.path.exists(VECTOR_DB_PATH):
        st.error(f"🚨 HATA: Veritabanı yok. Lütfen önce terminalde 'python ingest.py' komutunu çalıştırın.")
        return None
    
    # 1. Vektör Veritabanını geri yükle
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)
    
    # RAG KAPSAMINI GENİŞLETTİK (k=5): En alakalı 5 metin parçasını çek.
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) 
    
    # 2. LLM ve Gelişmiş Türkçe Prompt
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.1)

    # ESNEK PROMPT: Bağlam yetersizse LLM'in genel bilgisini kullanmasını sağlıyor.
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
    
    # 3. LCEL ile RAG Zincirini Kurma
    rag_chain = (
        {"context": retriever | (lambda x: "\n\n".join([doc.page_content for doc in x])), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain
    
# Streamlit Ana Fonksiyon
def main():
    st.set_page_config(page_title="EE Ders Asistanı", layout="wide")
    
    # -------------------------------------------------------------
    # GÜNCELLENMİŞ BAŞLIK VE AÇIKLAMA (Son Hali)
    # -------------------------------------------------------------
    st.title("💡 Elektrik-Elektronik Mühendisliği Asistanı (RAG)")
    st.caption("🔍 Hacettepe EE Ders Kitapları ve Yardımcı Kaynaklara Dayalı, Gemini Destekli Akıllı Asistan.")
    # -------------------------------------------------------------
    
    if 'rag_chain' not in st.session_state:
        st.session_state.rag_chain = setup_rag_chain()
        if st.session_state.rag_chain is None:
            return
    
    user_question = st.text_input("EE Ders Kitaplarına dair teknik bir soru sor:", 
                                  placeholder="Örn: 'ELE320 dersinden Dış Kaynaktan Beslenen BJT'nin DC Analizini açıklar mısın?'")
    
    if user_question:
        with st.spinner("🚀 Bilgi Kaynakların Taranıyor..."):
            try:
                response = st.session_state.rag_chain.invoke(user_question)
                st.success("Cevap Hazır!")
                st.markdown(f"**Asistan Yanıtı:** \n\n {response}")
            except Exception as e:
                st.error(f"Hata Kodu: {e}")
                st.warning("Uygulama açıldıysa ve bu hatayı alıyorsanız: günlük kota dolmuş olabilir veya API anahtarınız Google Cloud'da kısıtlanmıştır.")
    
if __name__ == "__main__":
    main()

# 🏃 Çalıştırma Komutu: streamlit run app.py