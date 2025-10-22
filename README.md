💡Elektrik-Elektronik Mühendisliği Asistanı: RAG Mimarisi

Proje Adı: Akbank GenAI Bootcamp Bitirme Projesi

Geliştirici: Ufuk Baran Karadağ (Hacettepe Üniversitesi, EEM 4. Sınıf)

Bu proje, Akbank GenAI Bootcamp kapsamında, Retrieval Augmented Generation (RAG) mimarisini kullanarak, mühendislik öğrencilerine özel bir bilgi asistanı geliştirmek amacıyla oluşturulmuştur.

🎯 Projenin Amacı
Projenin temel amacı, Hacettepe Üniversitesi Elektrik-Elektronik Mühendisliği ders müfredatını oluşturan İngilizce teknik kaynakları (ders kitapları, notlar vb.), yapay zeka aracılığıyla işleyerek, kullanıcının sorduğu spesifik teknik sorulara doğrulanmış ve detaylı Türkçe cevaplar üreten bir web tabanlı sohbet robotu geliştirmektir. Bu sayede, öğrenciler İngilizce kaynaklardaki bilgilere hızlı ve pratik bir şekilde ulaşabilir.

📚 Veri Seti Hakkında Bilgi
Projenin gücünü oluşturan veri tabanı, hazır veri setlerinden değil, direkt olarak mühendislik eğitimine özgü kaynaklardan oluşturulmuştur:
Veri Tipi ve Kapsam: EE Mühendisliği müfredatındaki ana dersleri (Analog Elektronik, Sayısal Tasarım, Kontrol Sistemleri, Makine Öğrenmesi, Sinyal ve Sistemler vb.) kapsayan 381 adet ders kitabı ve notu (PDF) kullanılmıştır.
Hacim: Toplamda 16.000'den fazla sayfa işlenmiş ve 38.000'den fazla bilgi parçasına ayrılmıştır.
Hazırlık Metodolojisi: Veri seti, yerel data/ klasörüne yüklenmiş ve LangChain'in DirectoryLoader özelliği ile toplu olarak okunmuş, işlenmiştir. Bu metodoloji, büyük ölçekli ve çeşitli dosya formatlarından veri alımını başarıyla yönetmiştir.

⚙️ Çözüm Mimarisi ve Kullanılan Yöntemler
Proje, güncel GenAI teknolojilerini bir araya getiren modern bir RAG (Retrieval Augmented Generation) mimarisi üzerine kurulmuştur.
LLM (Generation Model): Google Gemini API (gemini-2.5-flash) -Kullanıcı sorusuna bağlama dayalı nihai cevabı üretmek-
Embedding Model: Google GenerativeAI Embeddings (embedding-001) -Metin parçalarını vektörlere dönüştürmek-
Vektör Veritabanı: ChromaDB -Vektörleri depolamak ve hızlı, anlamsal arama yapmak-
RAG Çatısı: LangChain (LCEL) -Tüm RAG adımlarını (yükleme, bölme, arama, prompt'lama) zincirlemek-
Web Arayüzü: Streamlit -Kullanıcı dostu ve hızlı bir arayüz sunmak-

Özel Vurgu: Esnek Prompt Mühendisliği: Cevap alamama riskini ortadan kaldırmak için, LLM'e "Bağlamda net bir cevap yoksa, kendi genel mühendislik bilgini kullan" talimatı verilmiştir. Bu, RAG'ın doğruluğunu korurken, chatbot'un pratik kullanılabilirliğini artırmıştır.

📈 Elde Edilen Sonuçlar
Proje, yalnızca temel bilgileri değil, aynı zamanda mühendislik eğitimine özgü karmaşık konuları da başarıyla yanıtlayabildiğini kanıtlamıştır:
Derin Cevaplar: MOSFET, BJT modelleri, Kontrol Sistemleri Bode diyagramları ve Makine Öğrenmesi (L1/L2) gibi yüksek seviye teknik sorulara, kaynaklara dayalı, yapılandırılmış ve detaylı Türkçe cevaplar üretilmiştir.
Ölçeklenebilirlik: 381 adet dosyanın 18 dakikada başarılı bir şekilde işlenmesi, kodun endüstriyel ölçekte veri yönetimine uygun olduğunu göstermiştir.
Profesyonel Sunum: Çözüm, enerji dolu, profesyonel (Z kuşağına uygun) bir tonda cevaplar üretmektedir.

⚙️ Kodunuzun Çalışma Kılavuzu (Local Ortam)
Projeyi yerel olarak çalıştırmak için aşağıdaki adımları sırasıyla uygulayınız.

1.Repo Klonlama: Repoyu yerel bilgisayarınıza indirin.
git clone https://github.com/UfukBaranK/Akbank-GenAI-EE-Asistan
cd [Akbank GenAI Projesi]

2.Sanal Ortam Kurulumu:
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate

3.Bağımlılıkların Kurulumu:
pip install -r requirements.txt

4.API Anahtarını Tanımlama:
Proje klasörüne .env adında bir dosya oluşturun.
İçine Gemini API anahtarınızı tanımlayın: GEMINI_API_KEY="AIzaSyA_XXXXXXXXXXXXXXXXXXXXX"

5.Veri Seti Hazırlığı (Dış Kaynak):
rojenin çalışması için gerekli PDF/kaynak dosyalarını data/ klasörüne yerleştirin.
(Not: Telif hakkı nedeniyle 381 adet PDF repoda yer almamaktadır.)

6. Vektör Veritabanını Oluşturma:
Bu komut, tüm PDF'leri okur, parçalara ayırır ve vector_db/ klasörünü oluşturur:
python ingest.py

7.Uygulamayı Başlatma:
streamlit run app.py

🌐 Web Arayüzü & Product Kılavuzu
Bu bölüm, uygulamanın çalıştırılması, kabiliyetleri ve erişim detaylarını açıklamaktadır.
Erişim Bilgileri (Local URL)
Uygulama, yerel ortamınızda başarıyla başlatılmıştır. Erişim için aşağıdaki URL'leri kullanabilirsiniz:
Yerel URL (Kendi Bilgisayarınızdan Erişim): http://localhost:8501
Ağ URL'si (Aynı Yerel Ağdaki Cihazlardan Erişim): http://192.168.1.6:8501
(Not: Global erişim için mutlaka Streamlit Community Cloud gibi bir platforma dağıtılmalıdır.)

https://akbank-genai-ee-asistan-gtkdpyindmkcup2rcaxxzy.streamlit.app/
⚠️ NOT: Bu link, bulut ortamında 381 Kaynaklık RAG Kabiliyetini sergileyemez. Projenin tam kapasite çalıştığını test etmek için, lütfen README.md'deki Çalışma Kılavuzunu takip ederek projeyi yerel bilgisayarınızda (Network URL: http://192.168.1.6:8501) test ediniz.
