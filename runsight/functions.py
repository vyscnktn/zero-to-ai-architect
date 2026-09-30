"""
EndomondoHR (FitRec) koşu sınıflandırması — aerobik / anaerobik.

Veri yapısı varsayımı: her satır bir antrenman, her sütun 500 noktaya
yeniden örneklenmiş bir zaman serisi (timestamp, speed, heart_rate,
latitude, longitude, altitude). timestamp unix saniye.

Kaynak: https://cseweb.ucsd.edu/~jmcauley/datasets/fitrec.html

Pipeline (bkz. run_zone_pipeline): ham CSV -> QC -> özellik çıkarımı ->
kişisel nabız-bölgesi eşiği -> aerobik/anaerobik etiketi. Kümeleme
(GMM/K-means) YOK — long/easy run ayrımı zaten süre sütununa bakmakla
trivial, asıl anlamlı ayrım (eşik-altı/eşik-üstü efor, Seiler'in
polarized training modeli) doğrudan kişisel nabız eşiğinden hesaplanıyor.
"""

import ast
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# 1. Yükleme
# ----------------------------------------------------------------------


def parse_arr(s):
    if isinstance(s, np.ndarray):
        return s.astype(np.float64)
    cleaned = s.replace("[", "").replace("]", "").replace(",", " ")
    return np.fromstring(cleaned, sep=" ", dtype=np.float64)



# ----------------------------------------------------------------------
# 2. Satır bazlı kalite kontrolü
# ----------------------------------------------------------------------

QC_RULES = {
    "min_points": 100,        # 500'den az nokta kalmışsa seri güvenilmez
    "max_nan_frac": 0.20,     # %20'den fazla eksik varsa at
    "min_duration_min": 10.0,
    "max_duration_min": 300.0,
    "speed_valid": (1.0, 30.0),      # km/h — bunun dışı GPS hatası
    "hr_valid": (60.0, 220.0),       # bpm
    "min_hr_frac": 0.50,      # HR'ın en az yarısı geçerli aralıkta olmalı
}


def qc_row(row):
    """Bir antrenmanı değerlendir. (gecerli_mi, sebep, temiz_seriler) döner."""
    ts = row.get("timestamp")
    spd = row.get("speed")
    hr = row.get("heart_rate")


    duration_min = (ts[-1] - ts[0]) / 60.0
    if not np.isfinite(duration_min):
        return False, "bozuk_timestamp", None
    if duration_min < QC_RULES["min_duration_min"]:
        return False, "cok_kisa", None
    if duration_min > QC_RULES["max_duration_min"]:
        return False, "cok_uzun", None

    # Geçersiz değerleri NaN yap (silmek yerine — zaman hizası bozulmasın)
    lo, hi = QC_RULES["speed_valid"]
    spd = np.where((spd >= lo) & (spd <= hi) & np.isfinite(spd), spd, np.nan)
    lo, hi = QC_RULES["hr_valid"]
    hr = np.where((hr >= lo) & (hr <= hi) & np.isfinite(hr), hr, np.nan)

    spd_nan = np.isnan(spd).mean()
    hr_nan = np.isnan(hr).mean()
    if spd_nan > QC_RULES["max_nan_frac"]:
        return False, "hiz_cok_eksik", None
    if hr_nan > (1.0 - QC_RULES["min_hr_frac"]):
        return False, "hr_cok_eksik", None

    # Kalan boşlukları doğrusal doldur (zaman serisi, komşu nokta iyi tahmin)
    spd = _interp_nan(spd)
    hr = _interp_nan(hr)

    return True, "ok", {"ts": ts, "speed": spd, "hr": hr,
                        "duration_min": duration_min,
                        "nan_frac": max(spd_nan, hr_nan)}


def _interp_nan(arr):
    """NaN'ları komşularından doğrusal interpolasyonla doldur."""
    nans = np.isnan(arr)

    idx = np.arange(len(arr))
    out = arr.copy()
    out[nans] = np.interp(idx[nans], idx[~nans], arr[~nans])
    return out

# ----------------------------------------------------------------------
# 3. Özellikler
# ----------------------------------------------------------------------

def extract_features(series):
    """
    Bir antrenmandan, süreden bağımsız ve yorumlanabilir özellikler çıkar.

    Önemli: 500 nokta, antrenmanın TAMAMINA yayılmış. Yani 2 saatlik koşuda
    noktalar ~14 sn arayla, 30 dakikalıkta ~3.6 sn arayla. Bu yüzden
    "ardışık noktalar arası fark" türü ölçüler süreye bağlı çıkar ve
    antrenmanlar arasında karşılaştırılamaz. Aşağıdaki ölçüler ya
    ölçekten bağımsız (CV, oran) ya da zamana normalize edilmiş.
    """
    ts = series["ts"]
    spd = series["speed"]
    hr = series["hr"]
    dur = series["duration_min"]

    dt = np.diff(ts)                          # saniye
    dist_km = float(np.sum(spd[:-1] * dt) / 3600.0)

    med_spd = float(np.median(spd))
    p10, p90 = np.percentile(spd, [10, 90])

    # Tempo değişkenliği — ölçek bağımsız
    speed_cv = float(np.std(spd) / med_spd) if med_spd > 0 else 0.0
    speed_spread = float((p90 - p10) / med_spd) if med_spd > 0 else 0.0

    # Interval imzası: hız kaç kez kendi medyanının %15 üstüne çıkıp indi?
    # Dakikaya normalize → süreden bağımsız
    surges = _count_surges(spd, threshold=med_spd * 1.15)
    surge_rate = surges / dur if dur > 0 else 0.0

    # Long run imzası: son çeyrek ilk çeyreğe göre ne kadar yavaşladı,
    # nabız ne kadar yükseldi (kardiyak drift)
    q = max(len(spd) // 4, 1)
    pace_drift = float(np.mean(spd[-q:]) / np.mean(spd[:q])) if np.mean(spd[:q]) > 0 else 1.0
    hr_drift = float(np.mean(hr[-q:]) - np.mean(hr[:q]))

    return {
        "duration_min": dur,
        "distance_km": dist_km,
        "speed_med": med_spd,
        "speed_cv": speed_cv,
        "speed_spread": speed_spread,
        "surge_rate": surge_rate,
        "pace_drift": pace_drift,
        "hr_med": float(np.median(hr)),
        "hr_p10": float(np.percentile(hr, 10)),
        "hr_p95": float(np.percentile(hr, 95)),
        "hr_drift": hr_drift,
        # metadata — kümelemeye GİRMEZ, sadece filtre/rapor için
        "_nan_frac": series["nan_frac"],
    }


def _count_surges(x, threshold, min_len=3):
    """Sinyalin eşiğin üstünde geçirdiği ayrı blokların sayısı."""
    above = x > threshold
    if not above.any():
        return 0
    # Blok başlangıçlarını bul
    edges = np.diff(above.astype(int))
    starts = np.where(edges == 1)[0] + 1
    if above[0]:
        starts = np.r_[0, starts]
    ends = np.where(edges == -1)[0] + 1
    if above[-1]:
        ends = np.r_[ends, len(x)]
    return int(np.sum((ends - starts) >= min_len))


def _run_lengths_above(mask, min_len=3):
    """
    Boolean maskede eşiğin üstündeki ayrı blokların uzunluk listesi
    (örnek sayısı cinsinden). tempo TEK UZUN blok, interval ÇOK SAYIDA
    KISA blok üretir — add_hr_zone_features bu ayrımı burada kullanır.
    """
    if not mask.any():
        return []
    edges = np.diff(mask.astype(int))
    starts = np.where(edges == 1)[0] + 1
    if mask[0]:
        starts = np.r_[0, starts]
    ends = np.where(edges == -1)[0] + 1
    if mask[-1]:
        ends = np.r_[ends, len(mask)]
    lengths = ends - starts
    return lengths[lengths >= min_len].tolist()


# ----------------------------------------------------------------------
# 5. Kişiye göre normalizasyon
# ----------------------------------------------------------------------

PERSONAL_COLS = ["speed_med", "hr_med", "hr_p95"]


def add_personal_features(X, user_ids, min_workouts=5):
    """
    Aynı 11 km/h, iyi bir koşucu için kolay tempo, yeni başlayan için
    yarış temposudur. Mutlak hız/nabız kişiler arası karşılaştırılamaz.
    Her ölçüyü kişinin kendi medyanına oranla.

    Yeterli antrenmanı olmayan kişiler için genel medyan kullanılır ve
    _personal_ref=False ile işaretlenir.
    """
    X = X.copy()
    X["_user"] = np.asarray(user_ids)

    counts = X["_user"].value_counts()
    enough = set(counts[counts >= min_workouts].index)
    X["_personal_ref"] = X["_user"].isin(enough)

    for col in PERSONAL_COLS:
        global_ref = X[col].median()
        user_ref = X.groupby("_user")[col].transform("median")
        ref = np.where(X["_personal_ref"], user_ref, global_ref)
        ref = np.where(np.asarray(ref) > 0, ref, global_ref)
        X[f"rel_{col}"] = X[col] / ref

    return X


# ----------------------------------------------------------------------
# 5b. Kişiselleştirilmiş nabız-bölgesi özellikleri
# ----------------------------------------------------------------------
#
# Neden burada: yaş/HRmax formülü (220-yaş vb.) yok — veri setinde yaş
# bilgisi yok. Laktat eşiğini mutlak olarak bilemeyiz. Ama kişinin
# TÜM antrenmanları boyunca gözlemlenen nabız aralığını referans alabiliriz
# — Karvonen'in Heart Rate Reserve mantığının veri-temelli, kişiselleştirilmiş
# hali. Bu, yaş verisinin yokluğunu dezavantaj olmaktan çıkarıp veri-temelli
# bir avantaja çeviriyor: nüfus ortalaması bir formül yerine, kişinin kendi
# gerçek efor tarihçesini kullanıyoruz.
#
# Ayırt edici mantık:
#   - easy/long run: eşiğin üstünde neredeyse hiç zaman yok
#   - tempo/threshold: eşiğin üstünde TEK UZUN blok (sürdürülen efor)
#   - interval: eşiğin üstünde ÇOK SAYIDA KISA blok (tekrarlı efor)

def add_hr_zone_features(X, series_list, threshold_frac=0.70, min_workouts=5):
    """
    X'e nabız-bölgesi özellikleri ekler. series_list, X ile AYNI SIRADA
    olmalı (prepare_features içinde add_personal_features'tan hemen sonra
    çağrılır — series_list halen bellekte).

    threshold_frac=0.70: bu, AEROBİK EŞİK (VT1) — "eşik-altı/eşik-üstü"
    ikili ayrımı (Seiler'in polarized training modeli) için doğru sınır.
    Daha yüksek bir değer (örn. 0.85) LAKTAT EŞİĞİNE (VT2) karşılık gelir
    ve sadece interval-seviyesi zirveleri yakalar — tempo/threshold
    efor (genelde %70-80 HRR, sürdürülen) bunun altında kalıp yanlışlıkla
    "aerobik" görünür (sentetik testte tam bu hata çıktı, bkz.
    test_hr_zones.py).

    personal_hr_ceiling: kişinin TÜM antrenmanlarındaki en yüksek p95 nabzı
                         — "bu kişinin verideki en sert efor" proxy'si
    personal_hr_floor:   kişinin TÜM antrenmanlarındaki en düşük p10 nabzı
                         — "bu kişinin dinlenmeye yakın efor" proxy'si
    threshold = floor + threshold_frac * (ceiling - floor)
    """
    X = X.copy()

    ceiling = X.groupby("_user")["hr_p95"].transform("max")
    floor = X.groupby("_user")["hr_p10"].transform("min")

    counts = X["_user"].value_counts()
    enough = X["_user"].isin(counts[counts >= min_workouts].index)
    global_ceiling = X["hr_p95"].max()
    global_floor = X["hr_p10"].min()
    ceiling = np.where(enough, ceiling, global_ceiling)
    floor = np.where(enough, floor, global_floor)

    hr_range = np.maximum(ceiling - floor, 1e-6)
    threshold = floor + threshold_frac * hr_range
    X["hr_reserve_frac"] = np.clip((X["hr_p95"].to_numpy() - floor) / hr_range, 0, 1)

    time_above, excursion_rate, max_excursion_min = [], [], []
    for pos, (thr, s) in enumerate(zip(threshold, series_list)):
        hr = s["hr"]
        above = hr > thr
        n_samples = len(hr)
        dur = s["duration_min"]

        time_above.append(float(above.mean()))

        lengths = _run_lengths_above(above, min_len=3)
        excursion_rate.append(len(lengths) / dur if dur > 0 else 0.0)
        if lengths:
            max_len_frac = max(lengths) / n_samples
            max_excursion_min.append(max_len_frac * dur)
        else:
            max_excursion_min.append(0.0)

    X["hr_time_above_threshold"] = time_above
    X["hr_excursion_rate"] = excursion_rate
    X["hr_max_excursion_min"] = max_excursion_min

    return X


# ----------------------------------------------------------------------
# 6. Ortak hazırlık — yükleme sonrası, etiketleme öncesi
# ----------------------------------------------------------------------

def prepare_features(df, user_col="userId", min_n=20, verbose=True):
    """
    Ham df -> özellik matrisi X + series_list (ham, temiz zaman serileri).

    Adımlar: spor filtresi -> QC -> özellik çıkarımı -> kişisel
    normalizasyon -> nabız-bölgesi özellikleri.
    """
    log = (lambda *a: print(*a)) if verbose else (lambda *a: None)
    log(f"Ham veri: {len(df)} antrenman")


    log(f"  'sport' sütununa göre koşu: {len(df)}")

    idx, series_list, reasons = [], [], []
    for i, row in df.iterrows():
        ok, reason, s = qc_row(row)
        reasons.append(reason)
        if ok:
            idx.append(i)
            series_list.append(s)
    log("  QC:", pd.Series(reasons).value_counts().to_dict())

    if len(idx) < min_n:
        raise ValueError(f"Çok az antrenman kaldı: {len(idx)}")

    X = pd.DataFrame([extract_features(s) for s in series_list], index=idx)
    users = df.loc[idx, user_col].to_numpy() if user_col in df.columns else np.zeros(len(idx))
    X = add_personal_features(X, users)

    X = add_hr_zone_features(X, series_list)   # series_list ile AYNI SIRADA olmalı
    log(f"  özellik matrisi: {X.shape}")
    log(f"  kişisel referansı olan: {X['_personal_ref'].mean():.0%}")

    return X, series_list, idx


# ----------------------------------------------------------------------
# 7. Aerobik / anaerobik — doğrudan hesaplanan ikili etiket
# ----------------------------------------------------------------------
#
# Long run / easy run ayrımı zaten süre sütununa BAKMAKLA trivial —
# bunun için kümeleme gerekmiyor. Asıl anlamlı ayrım eşik-altı/eşik-üstü
# efor (Seiler'in polarized training modeli) ve bunun için elimizde
# zaten fizyolojik gerekçeli bir sinyal var: hr_time_above_threshold
# (kişinin kendi nabız tarihçesine göre eşiğin üstünde geçen süre oranı).

def add_training_zone(X, cutoff=0.02):
    """
    'aerobik' / 'anaerobik' ikili etiketi + sürekli skor ekler.

    anaerobic_time_frac: hr_time_above_threshold'un takma adı — sürenin
    ne kadarı kişisel eşiğin üstünde geçti. Sürekli skor olarak SAKLA;
    "easy run'da sprint" gibi karışık antrenmanlarda tek bir sınıfa
    zorlamak yerine gerçek oranı taşıyor (0.05 = %95 aerobik + %5
    anaerobik, ne "aerobik" ne "anaerobik" demek zorunda değilsin,
    ama istersen cutoff ile ikili karara da çevirebilirsin).

    cutoff: sentetik testte saf easy/long run'lar tam 0.0 çıktı (bkz.
    test_hr_zones.py); 0.02 gürültüden kaynaklı yanlış pozitife karşı
    küçük bir pay. Kendi verinde `training_zone_histogram` ile
    dağılımı gör, sıfırdaki yığılmanın nerede bittiğine göre ayarla.
    """
    X = X.copy()
    X["anaerobic_time_frac"] = X["hr_time_above_threshold"]
    X["training_zone"] = np.where(X["anaerobic_time_frac"] > cutoff,
                                  "anaerobik", "aerobik")
    return X


def run_zone_pipeline(df, cutoff=0.02, user_col="userId", verbose=True):
    """
    Ham df -> her antrenman için anaerobic_time_frac + training_zone.

    Adımlar: QC + kişisel nabız-bölgesi eşiği + eşik uygulaması.
    Kümeleme (GMM/K-means/K seçimi) YOK.
    """
    X, series_list, idx = prepare_features(df, user_col=user_col, verbose=verbose)
    X = add_training_zone(X, cutoff=cutoff)

    log = (lambda *a: print(*a)) if verbose else (lambda *a: None)
    counts = X["training_zone"].value_counts()
    log(f"  training_zone: {counts.to_dict()}")

    return X, {"series_list": series_list, "index": idx, "cutoff": cutoff}
