# RunSight — AI Koşu Koçu

Strava'dan çekilen antrenman verisini bir **GRU derin öğrenme modeli** ile aerobik/anaerobik olarak sınıflandırıp, bir **LLM ajanının** (RAG destekli, araç kullanan) kişiselleştirilmiş koçluk raporuna dönüştüren, uçtan uca **n8n** otomasyonu.

> Bootcamp bitirme projesi. Pipeline: **DL modeli → LLM Ajanı → n8n otomasyonu**.

---

## 1. Problem

Koşucular genelde antrenmanlarının "ne işe yaradığını" bilmeden koşar: bir tempo koşusu gerçekten hedeflenen yoğunlukta mı geçti, yoksa farkında olmadan aerobik tabanı mı aşındı (ya da tam tersi, "kolay koşu" günü aslında anaerobik bölgede mi geçti)? Spor bilimi literatüründe bu ayrım **polarized training** (Seiler, 2010) modelinin temelini oluşturuyor: elit dayanıklılık sporcularının antrenman hacminin büyük kısmı düşük yoğunlukta (aerobik eşiğin altında), küçük bir kısmı ise yüksek yoğunlukta (eşiğin belirgin üstünde) geçmeli — ortadaki "gri bölge" (tempo/threshold) fazla kullanılırsa performans platoya girer.

RunSight bu ayrımı otomatikleştirip her antrenman sonrası koşucuya **veriye dayalı, kişiselleştirilmiş bir geri bildirim** sunmayı hedefliyor:

- Antrenman gerçekten planlanan yoğunlukta mı geçti?
- Nabız driftı (aerobic decoupling) var mı, toparlanma nasıl?
- Sonraki antrenman için somut, 3 maddelik aksiyon önerisi ne olmalı?

## 2. Veri

**Kaynak:** [EndomondoHR / FitRec](https://cseweb.ucsd.edu/~jmcauley/datasets/fitrec.html) — halka açık, GPS+nabız içeren koşu/bisiklet antrenman veri seti. Her satır bir antrenman; her sütun, antrenmanın tamamına yayılmış 500 noktaya yeniden örneklenmiş bir zaman serisi (`timestamp`, `speed`, `heart_rate`, `latitude`, `longitude`, `altitude`).

Ham dosyalar Python-dict formatında JSON Lines olarak geliyor (`ast.literal_eval` ile parse ediliyor) — standart `json.loads` çalışmıyor çünkü tek tırnak ve `nan` gibi Python-özel literaller içeriyor.

### Kalite kontrolü (QC)

Her antrenman aşağıdaki kurallara göre elenip/temizleniyor (`functions.py::qc_row`):

| Kural | Eşik |
|---|---|
| Minimum süre | 10 dk |
| Maksimum süre | 300 dk |
| Geçerli hız aralığı | 1–30 km/h (dışı GPS hatası sayılır) |
| Geçerli nabız aralığı | 60–220 bpm |
| Maks. eksik hız verisi | %20 |
| Min. geçerli nabız oranı | %50 |

Geçersiz değerler silinmiyor, `NaN` yapılıp zaman serisinin hizası bozulmadan komşu noktalardan doğrusal interpolasyonla dolduruluyor.

### Özellik çıkarımı

500 nokta antrenmanın tamamına yayıldığı için (2 saatlik koşuda ~14 sn, 30 dakikalıkta ~3.6 sn aralıkla), "ardışık nokta farkı" gibi ölçüler süreye bağlı çıkıp antrenmanlar arası kıyaslanamaz hale gelir. Bunun yerine ölçekten bağımsız / zamana normalize edilmiş özellikler çıkarılıyor: tempo değişkenliği (CV), interval imzası (dakikaya normalize "surge" sayısı), long-run imzası (pace drift, HR drift, ilk çeyrek vs son çeyrek), nabız yüzdelikleri.

### Kişiye göre normalizasyon

Aynı 11 km/h, iyi bir koşucu için kolay tempo, yeni başlayan için yarış temposudur — mutlak hız/nabız kişiler arası kıyaslanamaz. Her koşucuunun kendi medyanına oranlanış (`rel_*`) özellikler ekleniyor; yeterli antrenmanı olmayan kullanıcılar için genel medyana düşülüyor (`_personal_ref=False` ile işaretlenerek).

## 3. Etiketleme — Kural Tabanlı Pipeline (Baseline)

Veri setinde yaş bilgisi yok, dolayısıyla klasik `220-yaş` formülüyle HRmax hesaplanamıyor; laktat eşiği de doğrudan ölçülmüyor. Bunun yerine **kişinin kendi antrenman geçmişinde gözlenen nabız aralığı** referans alınıyor — Karvonen'in Heart Rate Reserve mantığının veri-temelli, kişiselleştirilmiş hali:

```
personal_hr_ceiling = kişinin tüm antrenmanlarındaki en yüksek p95 nabız
personal_hr_floor   = kişinin tüm antrenmanlarındaki en düşük p10 nabız
threshold = floor + 0.70 × (ceiling − floor)
```

`threshold_frac = 0.70` **aerobik eşiğe (VT1)** karşılık geliyor — "eşik-altı / eşik-üstü" ikili ayrımı (Seiler'in polarized training modeli) için doğru sınır. Daha yüksek bir değer (örn. 0.85, laktat eşiği/VT2) sadece interval-seviyesi zirveleri yakalar; sürdürülen tempo/threshold efor (genelde %70-80 HRR) bunun altında kalıp yanlışlıkla "aerobik" görünür (sentetik testte bu hata gözlemlendi).

Her antrenman için nabzın eşiğin üstünde geçirdiği süre oranı (`anaerobic_time_frac`) hesaplanıyor ve `cutoff=0.02` ile ikili etikete çevriliyor:

```python
training_zone = "anaerobik" if anaerobic_time_frac > 0.02 else "aerobik"
```

**Not — kümeleme kullanılmadı:** İlk yaklaşımda K-Means/GMM ile kümeleme denendi, ancak terk edildi. Long/easy run ayrımı zaten süre sütununa bakmakla trivial; asıl anlamlı ayrım (eşik-altı/eşik-üstü efor) doğrudan fizyolojik gerekçeli bir sinyalden (kişisel nabız eşiği) hesaplanabiliyor — kümelemenin ekleyeceği bir değer yok.

Bu kural tabanlı pipeline (`run_zone_pipeline`) **projenin baseline'ı**: hem GRU modelinin eğitim etiketlerini üretiyor hem de canlı üründe (deploy edilen Lambda çökerse ya da karşılaştırma için) bağımsız bir referans noktası sağlıyor.

## 4. DL Modeli — GRU

### Neden GRU / neden kural tabanlı pipeline'ı taklit etmek?

Kural tabanlı pipeline zaten doğru ve yorumlanabilir bir etiket üretiyor — ama girdisi olarak **kişinin geçmiş antrenman tarihçesini** (personal HR ceiling/floor) gerektiriyor, yani soğuk başlangıçta (yeni kullanıcı, az veri) güvenilir çalışmıyor ve her tahminde tüm geçmişin yeniden hesaplanmasını istiyor.

GRU modeli, **aynı etiketi ham sinyalden (speed, heart_rate, altitude, GPS-türetilmiş konum) tek bir antrenmanın zaman serisine bakarak** tahmin etmeyi öğreniyor — kişisel eşik özelliğine görmeden erişimi yok. Bu yüzden karşılaştırma "hangisi daha doğru" değil, **distillation-fidelity**: GRU, kural tabanlı pipeline'ın kararını ham sinyalden ne kadar sadakatle yeniden üretebiliyor? Yüksek bir uyum, modelin kişisel-eşik mantığını sinyal içindeki örtüntülerden (tempo değişkenliği, nabız driftı, interval imzası vb.) örtük olarak öğrendiğini gösteriyor — ve üretimde tek-antrenmanlık, geçmişsiz tahmin imkânı sağlıyor.

### Girdi kanalları

Eğitim ve deploy'da (`8_gru_deploy.py::_extract_channels`) kullanılan 4 kanal, her biri 500 noktaya yeniden örneklenmiş:

1. `speed` (km/h)
2. `heart_rate` (bpm)
3. `altitude` (m)
4. `latlon_pca` — lat/lon önce yerel metre koordinatına çevriliyor (eşit-dikdörtgen yaklaşım), sonra en çok varyans taşıyan tek eksene (PCA, 1 bileşen) projekte ediliyor — GPS'in 2 boyutunu modele tek, anlamlı bir "konum değişimi" sinyaline indirgemek için.

### Ön işleme

- Her kanal için ayrı `MinMaxScaler`, **sadece eğitim setinde fit edilip** hem train hem test'e uygulanıyor (veri sızıntısını önlemek için).
- Scaler'lar her zaman adımını (500 tanesi) ayrı bir "özellik" olarak fit ediyor — inference'ta da aynı `(1, 500)` şekliyle uygulanmalı, yoksa scaler yanlış hizalanır.
- Eğitim/test bölünmesi **antrenman bazlı** yapıldı (kullanıcı bazlı değil) — aynı kullanıcının farklı antrenmanları train/test arasında sızabilir riskine karşı bilinçli bir tasarım tercihi; asıl korunan sızıntı ekseni özellik ölçekleme (scaler fit) tarafında.

### Mimari ve eğitim

- **GRU (Gated Recurrent Unit)** tabanlı dizi sınıflandırıcı.
- Hiperparametre araması `keras_tuner` ile yapıldı.
- Sınıf dengesizliğine karşı `class_weight` kullanıldı.
- Tuning hedefi: `val_auc`.
- Çıktı: sigmoid olasılık (anaerobik olasılığı) → `confidence = |proba − 0.5| × 2`.

## 5. LLM Ajanı

n8n içinde **LangChain Agent** node'u olarak çalışıyor, üç akışta:

- **Onboarding ajanı** — yeni kullanıcı profil/hedef bilgisini toplayıp yapılandırıyor.
- **Haftalık değerlendirme ajanı** — her pazar 20:00'de (cron `0 20 * * 0`) Notion'daki haftalık planı gerçekleşen antrenmanlarla karşılaştırıp uyum yüzdesi, öneriler ve sonraki hafta için küçük bir ayarlama üretiyor.
- **Antrenman sonrası koçluk ajanı** — GRU'nun sınıflandırmasını, geçmiş antrenman kaytlarını ve RAG bilgi tabanını kullanarak kişiselleştirilmiş rapor üretiyor.

### Sabit çıktı şeması (Structured Output Parser)

Ajan çıktısı, n8n'in Structured Output Parser node'una verilen manuel JSON Schema ile sabitleniyor — `summary`, `details` (gru_assessment, plan_comparison, load_and_recovery, data_quality, heart_rate_drift), tam olarak 3 elemanlı `action_items`, `uncertainties`, `sources_used`. Bu, downstream node'ların (Telegram raporu, Drive'a kaydedilen dosya) her zaman aynı alanlara güvenle erişebilmesini sağlıyor.

### Araçlar (Tools)

- Google Sheets'ten geçmiş antrenman kayıtlarını okuma.
- Notion'dan kullanıcı profil sayfasını okuma.
- **RAG bilgi tabanı** (bkz. aşağı) — spor bilimi kaynaklarından (Seiler 2010, laktat/eşik/HRmax metodolojisi dokümanları) alakalı pasajları getirme.

### Dinamik sistem promptları

Üç ajanın sistem promptu, n8n'in kendisine gömülü değil — çalışma zamanında GitHub reposundan (`raw.githubusercontent.com/.../post_workout_prompt.md`, `onboarding_prompt.md`, `weekly_review_prompt.md`) bir HTTP Request node ile çekiliyor. Bu sayede prompt değişikliği n8n'e girmeden, sadece repoya push ile yayılıyor.

### RAG (bonus)

**Pinecone** (serverless, 1536 boyut, cosine) tabanlı kalıcı bir RAG hattı:

1. **Insert:** Yüklenen spor bilimi dokümanları (Seiler 2010, laktat ölçümü, fonksiyonel eşik, maksimum nabız kullanımı üzerine kaynaklar) `Default Data Loader` ile parçalanıp `Embeddings OpenAI` ile vektörleştirilip Pinecone index'ine yazılıyor.
2. **Retrieve:** Aynı embedding modeliyle (tutarlılık için insert/retrieve'de zorunlu), ajana `knowledge_base` adında bir "retrieve-as-tool" aracı olarak bağlanıyor - ajan gerektiğinde kendi sorgusunu üretip alakalı pasajları çekiyor.

## 6. Confidence Bazlı Routing + İnsan Onayı (Human-in-the-loop, bonus)

GRU'nun `confidence` skoru (`|proba - 0.5| * 2`) **0.6** eşiğiyle dallanıyor:

- **Confidence >= 0.6:** GRU'nun tahmini doğrudan kabul edilip kayda geçiyor.
- **Confidence < 0.6:** Kullanıcıya Telegram üzerinden bir form gönderiliyor (`sendAndWait`), antrenmanın gerçekte aerobik mi anaerobik mi olduğunu kendisi işaretliyor; bu **düzeltilmiş** etiket (GRU'nun orijinal tahmini değil) kayda geçiyor.

Bu, modelin emin olmadığı durumlarda sessizce yanlış karar vermesini önleyip, aynı zamanda gelecekte fine-tuning için insan-doğrulamalı bir etiket akışı oluşturuyor.

## 7. n8n Otomasyonu — Mimari Şeması

Asağıdaki diyagram, workflow'nu üç ana akış halinde özetliyor:

**İ. Antrenman Sonrası Akış**

`Strava Trigger` → `Get Activity Streams` → `If (type == Run)` → `GRU Model API (AWS Lambda, ECR image)` → **`kardiyak kaymayi hesapla` (Code, JavaScript)**: GRU'nun döndürdüğü alanları (`training_zone`, `confidence`, `hr_p10`, `hr_p95`) korur, `Get Activity Streams` çıktısındaki ham `heartrate`/`velocity_smooth` dizilerinden **Efficiency Factor** (hız/nabız) yöntemiyle `hr_drift_pct` hesaplar: antrenman iki eşit yarıya bölünür, her yarı için ortalama hız/ortalama nabız oranı çıkarılır, iki yarı arasındaki yüzde fark raporlanır (pozitif = ikinci yarıda aynı nabızla daha az hız üretiliyor → kardiyak drift). Oran tabanlı olduğu için pace'in kasıtlı değiştiği (progression/interval) antrenmanlarda yanlış pozitifi büyük ölçüde önler; yetersiz veri veya çok düşük ortalama hızda (`< 3 km/h`, ör. ısınma ağırlıklı kayıt) `null` döner. → `Get row(s) in sheet` (geçmiş kayıt) → `If (confidence >= 0.6)`:

- True → `Append row (sheet)`
- False → `Telegram form` (insan onayı) → `Append row (sheet, düzeltilmiş etiket)`

Devam: `HTTP Request` (GitHub'dan `post_workout_prompt.md` çek) → `AI Agent` (Structured Output Parser + Sheets/Notion/RAG araçları) → `Create file` (Drive log) & `Telegram` (rapor).

**Antrenmanın plana göre hafta/gün konumu:** `athlete_id'yi al ve kaydet` node'u onboarding'de `plan_start_date` (`{{ $now }}`, tam ISO datetime) sütununu users sheet'ine yazıyor. Post-workout tarafında `antrenman günlerini hesapla` (Code) node'u bu tarihi antrenman tarihiyle kıyaslayıp `plan_hafta_no` ve `plan_gun_adi` alanlarını deterministik olarak hesaplıyor ve agent prompt'una geçiriyor — böylece LLM'in kendi başına tarih aritmetiği yapıp hata yapması engelleniyor, hangi antrenmanın planın kaçıncı haftasına/gününe denk geldiği n8n katmanında kesin olarak belirleniyor.

**II. Onboarding Akışı**

`Telegram Trigger` → ... → `Append/update row (sheet)` → `HTTP Request` (GitHub'dan `onboarding_prompt.md` çek) → `AI Agent` (Structured Output Parser + Gemini) → `Notion` sayfası oluştur → `Gmail` onayı → `Telegram` bildirimi.

**III. RAG Besleme (manuel tetik, kalıcı depo — Pinecone)**

`Dosya yükle` (PDF/MD, kaynak dokümanlar) → `Default Data Loader` → `Embeddings OpenAI` → `Pinecone Vector Store` (mode: insert) — index: `n8n` (Pinecone, serverless, dimension 1536, metric cosine).

`Pinecone Vector Store` (mode: retrieve-as-tool, `toolDescription`: "Use this knowledge base to answer questions from the user", aynı Embeddings modeli ile) → `AI Agent`'a "knowledge_base" aracı olarak bağlı. Kalıcı bir vektör veritabanı olduğu için n8n instance'ı yeniden başlasa da veri kaybolmuyor (önceki sürümde `vectorStoreInMemory` kullanılıyordu, bu artık Pinecone ile değiştirildi).

**IV. Haftalık Takip ve Öğrenme Döngüsü (bonus — İleri seviye)**

`Haftalık Tetikleyici` (Schedule Trigger, her Pazar 20:00 — cron `0 20 * * 0`) → `Tüm Kullanıcılar` (users sheet) → `Aktif Planı Olanlar` (aktif bir `training_plan`'ı olanları filtrele) → `Loop Over Items` (kullanıcı başına döngü):

- `Haftalık Antrenmanlar` (trainings sheet'ten `chat_id`'ye göre kullanıcının kayıtları) → `Haftalık Özet Hesapla` (Code): son 7 günün antrenmanlarını filtreler; toplam km, antrenman sayısı ve zone dağılımını (aerobik/anaerobik) hesaplar.
- `Antrenman Var mı?` (`antrenman_sayisi > 0`):
  - **False** → `Motivasyon Mesajı` (Telegram) — bu hafta hiç antrenman yoksa nazik bir hatırlatma gönderir.
  - **True** → `Planı Oku` (Notion, `getMarkdown`) → `Haftalık Değerlendirme` (AI Agent + Gemini + Structured Output Parser): planlanan programla gerçekleşeni karşılaştırır, 0-100 arası bir **uyum yüzdesi**, 2-3 somut öneri ve gelecek hafta için hacim/yoğunluk ayarlama önerisi üretir → `Haftalık Rapor Gönder` (Telegram) → `Haftalık Özet Kaydet` (Sheets, log — girdi/çıktı denetlenebilirlik için saklanıyor).

Bu akış, "İleri seviye" hedeflerinden **çoklu girdi/zaman içi karşılaştırma**, **geçmiş kayıtlarla trend analizi** ve **otomatik takip ve öğrenme döngüsü** kriterlerini karşılıyor: sistem haftalık olarak kendiliğinden tetikleniyor, geçmiş antrenman kayıtlarını planla karşılaştırıyor ve bir sonraki haftanın planına somut bir ayarlama öneriyor.

## 8. Model Deployment

GRU modeli **AWS Lambda** üzerinde, Docker (ECR) image olarak sunuluyor (`Dockerfile`, `8_gru_deploy.py`):

- Base image: `public.ecr.aws/lambda/python:3.13`.
- `libgomp` TensorFlow runtime bağımlılığı için ayrıca kuruluyor.
- Katman önbelleği için önce `requirements-deploy.txt` kopyalanıp kuruluyor, kod en son kopyalanıyor.
- Model (`gru_model.keras`) ve ön işleme nesneleri (`gru_preprocessing.pkl` — kanal sırası + eğitimde fit edilmiş scaler'lar) image'a gömülü; Lambda container'ı sıcak kaldığı sürece bir kez yüklenip her çağrıda tekrar yüklenmiyor.

**Girdi:** Strava'nin `GET /activities/{id}/streams?keys=heartrate,altitude,latlng,velocity_smooth` cevabı, olduğu gibi POST ediliyor.

**Çıktı:**
```json
{
  "training_zone": "aerobik" | "anaerobik",
  "anaerobic_probability": 0.83,
  "confidence": 0.66,
  "hr_p10": 128.0,
  "hr_p95": 170.0
}
```

## 9. Kurulum

### Gereksinimler

```bash
# Eğitim/geliştirme ortamı (notebook'lar)
pip install -r requirements.txt

# Sadece inference (Lambda deploy) — eğitimdekiyle birebir aynı sürümler,
# model/scaler'ların kaydedildiği ortamla uyumsuzluk yaşanmasın diye
pip install -r requirements-deploy.txt
```

### Veri hazırlama ve model eğitimi

Notebook'lar sırayla çalıştırılır:

1. `1_data_preprocess.ipynb` → `7_gru_tuner.ipynb` — ham CSV'den GRU modeline kadar tüm pipeline (QC, özellik çıkarımı, kural tabanlı etiketleme, GRU ön işleme, hiperparametre araması, eğitim, değerlendirme).

### Lambda deploy

```bash
docker build -t runsight-gru .
# ECR'a push + Lambda function'ı bu image'dan oluştur/güncelle
```

### n8n (self-hosted)

Proje, n8n Cloud deneme sürümünden kendi sunucuma (Hetzner VPS, Docker, reverse proxy + SSL, `https://n8n.vyscnktn.de`) taşındı. Cloud'a özgü bir şeye bağımlı değil; n8n Cloud'da da aynı şekilde çalışır.

**Sunucu ortam değişkenleri:**

```
N8N_HOST=n8n.vyscnktn.de
N8N_PROTOCOL=https
WEBHOOK_URL=https://n8n.vyscnktn.de/
N8N_EDITOR_BASE_URL=https://n8n.vyscnktn.de
N8N_PROXY_HOPS=1
N8N_ENCRYPTION_KEY=<sabit, yedekle>
GENERIC_TIMEZONE=Europe/Berlin
TZ=Europe/Berlin
```

**Kurulum adımları:**

1. `n8n/runsight_flow.json` dosyasını import et.
2. Kimlik bilgilerini (credentials) oluştur ve node'lara bağla: Strava OAuth2, Telegram Bot, Google Sheets / Drive / Gmail (tek Google OAuth client), Notion, OpenAI (Agent + Embeddings), Google Gemini, Pinecone. Self-hosted'da Cloud'daki hazır AI kredileri yoktur; OpenAI ve Gemini için kendi API anahtarların gerekir.
3. OAuth redirect URI'leri: Google Cloud'da `https://n8n.vyscnktn.de/rest/oauth2-credential/callback` ekle. Strava'da "Authorization Callback Domain" alanına yalnızca alan adını yaz (`n8n.vyscnktn.de`). Google uygulaması "Testing" modundaysa token'lar 7 günde dolar; kalıcı olması için uygulamayı yayınla (Publish app).
4. GitHub raw prompt URL'lerinin (`HTTP Request` node'ları) doğru repo/branch'e işaret ettiğinden emin ol (`onboarding_prompt.md`, `post_workout_prompt.md`, `weekly_review_prompt.md`).
5. RAG için: `Upload your file here` form trigger'ından kaynak dokümanları bir kez yükleyip Pinecone index'ini doldur.
6. Workflow'u **publish** et (n8n 2.x'te taslak kaydetmek production'ı değiştirmez; tetikleyiciler yalnızca yayınlanmış sürümle çalışır). Strava ve Telegram webhook'ları publish anında yeni alan adına kaydolur.

**Not — Python yok:** Self-hosted n8n'de Python runner bulunmaz, bu yüzden workflow'daki Code node'ları (Karvonen zonları, antrenman günü/hafta hesabı, kardiyak kayma) **JavaScript** ile yazıldı. Mantık Python sürümüyle birebir aynı; sayısal olarak doğrulandı (Karvonen zonları kullanıcı sheet'indeki değerlerle eşleşiyor).

## 10. Sonuçlar ve Kısıtlar

- Kural tabanlı pipeline (`run_zone_pipeline`), veri setindeki her antrenman için tutarlı, fizyolojik gerekçeli bir aerobik/anaerobik etiketi üretiyor ve GRU modelinin hem eğitim etiketi hem de değerlendirme referansı (distillation-fidelity) olarak kullanılıyor.
- GRU modeli, kişisel eşik bilgisine erişimi olmadan, sadece ham sinyalden bu etiketi yüksek sadakatle yeniden üretmeyi öğreniyor — bu da üründe tek-antrenmanlık, geçmişsiz (soğuk başlangıç) tahmini mümkün kılıyor.
- **Bilinen kısıt — çoklu kullanıcı (multi-tenant):** Strava webhook'u uygulama seviyesinde (client_id) tüm kullanıcılar için tetikleniyor, ama API çağrıları şu an geliştiricinin tek bir OAuth2 kimlik bilgisine bağlı. Üretime taşımak için kullanıcı başına OAuth token saklama/yenileme (per-athlete token store) eklenmesi gerekiyor — bilinçli olarak bu iterasyonun kapsamı dışında bırakıldı.
- **Çözüldü — kalıcı vektör store:** RAG başta `vectorStoreInMemory` ile kuruldu (n8n restart'ında sıfırlanıyordu); bu artık **Pinecone**'a taşındı (serverless index, dimension 1536, metric cosine) — veri kalıcı.
- **Bilinen kısıt — kişisel HR eşiği otomatik kalibrasyonu henüz yok:** Lambda API'sinin döndürdüğü `hr_p10`/`hr_p95` alanları, gelecekte kullanıcının `personal_hr_ceiling`/`personal_hr_floor` değerlerinin geçmiş antrenmanlar üzerinden otomatik yeniden hesaplanması için tasarlandı, ama bu adım haftalık öğrenme döngüsüne henüz entegre edilmedi — şu an döngü sadece plan-uyum karşılaştırması ve öneri üretiyor, eşik değerlerini güncellemiyor.
- **Bilinen kısıt — kardiyak drift metriği (`hr_drift_pct`) progressive/interval antrenmanlarda temkinli yorumlanmalı:** Efficiency Factor (hız/nabız) oranı, pace ile nabız arasındaki ilişkiyi eşik yakınında doğrusal varsayıyor. Steady-state/tempo koşularda güvenilir, ama kasıtlı olarak ikinci yarısı belirgin hızlandırılan (progression) antrenmanlarda fizyolojik olarak normal bir nabız artışını yanlışlıkla drift olarak işaretleyebilir. Agent prompt'unda bu durum için temkinli yorumlama talimatı var, ama algoritmik bir düzeltme (antrenman tipine göre ağırlıklandırma) henüz yapılmadı.

## 11. Proje Yapısı

```
runsight/
├── 1_data_preprocess.ipynb   # Ham veri yükleme, QC
├── 2..5_*.ipynb              # Özellik çıkarımı, kişisel normalizasyon, etiketleme
├── 6_gru_preprocess.ipynb    # GRU için kanal hazırlama, scaler fit
├── 7_gru_tuner.ipynb         # Hiperparametre araması, eğitim, değerlendirme
├── 8_gru_deploy.py           # Lambda handler (inference)
├── functions.py              # QC, özellik çıkarımı, kural tabanlı pipeline
├── Dockerfile                # Lambda/ECR image tanımı
├── requirements.txt          # Eğitim ortamı bağımlarıkları
├── requirements-deploy.txt   # Sadece inference bağımlılıkları
├── gru_model.keras           # Eğitilmiş model
├── gru_preprocessing.pkl     # Kanal sırası + scaler'lar
├── onboarding_prompt.md      # Onboarding ajanı sistem promptu
├── post_workout_prompt.md    # Antrenman sonrası koç ajanı sistem promptu
├── weekly_review_prompt.md   # Haftalık değerlendirme ajanı sistem promptu
├── schemas/                  # Ajan çıktı JSON şemaları (draft-07)
│   ├── onboarding_schema.json
│   ├── post_workout_schema.json
│   └── weekly_review_schema.json
├── n8n/
│   └── runsight_flow.json    # n8n workflow export (içe aktarılıp çalıştırılabilir)
└── rag_documents/            # RAG bilgi tabanı kaynakları (Seiler 2010, eşik/HRmax dokümanları)
```
