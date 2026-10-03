"""
AWS Lambda handler: Strava aktivite stream'inden aerobik/anaerobik tahmini.

Girdi (event["body"], JSON) — Strava'nin
GET /activities/{id}/streams?keys=heartrate,altitude,latlng,velocity_smooth
endpointinin dondurdugu format, oldugu gibi:

    {
      "heartrate":        {"data": [bpm, bpm, ...]},
      "altitude":         {"data": [m, m, ...]},
      "latlng":           {"data": [[lat, lon], [lat, lon], ...]},
      "velocity_smooth":  {"data": [m/s, m/s, ...]}
    }

n8n'in Strava API'den cektigi streams cevabini aynen bu endpoint'e POST
etmesi yeterli.

Cikti (JSON):
    {
      "training_zone": "aerobik" | "anaerobik",
      "anaerobic_probability": float,   # modelin sigmoid ciktisi (0-1)
      "confidence": float,              # |olasilik - 0.5| * 2  (0-1)
      "hr_p10": float | null,           # bu antrenmanin nabiz 10. yuzdeligi (bpm)
      "hr_p95": float | null            # bu antrenmanin nabiz 95. yuzdeligi (bpm)
    }

hr_p10/hr_p95: functions.py::add_hr_zone_features ile AYNI metodoloji
(np.percentile(hr, 10) / np.percentile(hr, 95)) — n8n tarafinda bu degerleri
antrenman basina loglayip, kullanicinin TUM antrenmanlari uzerinden
personal_hr_ceiling = max(hr_p95), personal_hr_floor = min(hr_p10) alinarak
kisisel esigin (threshold = floor + 0.70 * (ceiling - floor)) periyodik
olarak yeniden hesaplanmasini (otomatik kalibrasyon) mumkun kilmak icin
eklendi. Model/mimari degismedi, sadece zaten hesaplanan bir ara deger
disari aktarildi.

Bu dosyanin yaninda (ayni Docker image icinde) su iki dosya bulunmali:
    gru_model.keras         (6_gru_preprocess + 7_gru_tuner'in urettigi model)
    gru_preprocessing.pkl   (kanal sirasi + egitimde fit edilmis scaler'lar)
"""

import json
import pickle

import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = "gru_model.keras"
PREPROC_PATH = "gru_preprocessing.pkl"
N_POINTS = 500          # egitimdeki sabit dizi uzunlugu
R_EARTH = 6371000.0     # metre — lat/lon -> lokal koordinat donusumu icin

# Lambda container'i sicak kaldigi surece (warm invocation) bir kez yuklenir,
# her cagride tekrar degil — cold start disinda ek gecikme yaratmaz.
_model = load_model(MODEL_PATH)
with open(PREPROC_PATH, "rb") as f:
    _preproc = pickle.load(f)
_channel_order = _preproc["channel_order"]
_scalers = _preproc["scalers"]


def _resample_to_n(values, n=N_POINTS):
    """Strava stream'i (degisken uzunlukta) egitimdeki gibi sabit n noktaya indirger."""
    values = np.asarray(values, dtype=np.float64)
    old_idx = np.linspace(0.0, 1.0, num=len(values))
    new_idx = np.linspace(0.0, 1.0, num=n)
    return np.interp(new_idx, old_idx, values)


def _interp_nan(arr):
    """Strava sensor kopmalarindan (HR kayisi vb.) kalan NaN'lari komsularindan doldurur."""
    nan_mask = np.isnan(arr)
    if nan_mask.any():
        idx = np.arange(len(arr))
        arr[nan_mask] = np.interp(idx[nan_mask], idx[~nan_mask], arr[~nan_mask])
    return arr


def _latlon_to_pca(lat, lon):
    """Egitimdeki (6_gru_preprocess.ipynb) ile ayni donusum: once lokal metre
    koordinatina cevir (esitrektangular yaklasim), sonra en cok varyans tasiyan
    tek eksene (PCA, 1 bilesen) projekte et."""
    lat0_rad = np.radians(lat[0])
    x = R_EARTH * np.radians(lon - lon[0]) * np.cos(lat0_rad)
    y = R_EARTH * np.radians(lat - lat[0])

    xy = np.column_stack([x, y])
    xy = xy - xy.mean(axis=0)
    _, _, vt = np.linalg.svd(xy, full_matrices=False)
    return xy @ vt[0]


def _hr_percentiles(hr_raw):
    """functions.py::add_hr_zone_features ile AYNI metodoloji: bu antrenmanin
    nabiz 10./95. yuzdeligi. NaN'lar disarida birakilir; gecerli deger yoksa
    (None, None) doner."""
    hr_raw = np.asarray(hr_raw, dtype=np.float64)
    hr_raw = hr_raw[np.isfinite(hr_raw)]
    if hr_raw.size == 0:
        return None, None
    p10 = float(np.percentile(hr_raw, 10))
    p95 = float(np.percentile(hr_raw, 95))
    return p10, p95


def _extract_channels(streams):
    """Strava stream JSON'undan egitimdeki 4 kanali (speed, heart_rate, altitude,
    latlon_pca) cikarir ve her birini 500 noktaya indirger."""
    if "velocity_smooth" not in streams:
        raise ValueError(
            "velocity_smooth stream'i yok — Strava streams istegine "
            "'velocity_smooth' key'ini eklemen lazim (GPS'li aktivitelerde "
            "Strava bunu otomatik hesaplar)."
        )

    hr = np.asarray(streams["heartrate"]["data"], dtype=np.float64)
    alt = np.asarray(streams["altitude"]["data"], dtype=np.float64)
    latlng = np.asarray(streams["latlng"]["data"], dtype=np.float64)
    lat, lon = latlng[:, 0], latlng[:, 1]
    speed_kmh = np.asarray(streams["velocity_smooth"]["data"], dtype=np.float64) * 3.6

    raw = {
        "speed": speed_kmh,
        "heart_rate": hr,
        "altitude": alt,
        "latlon_pca": _latlon_to_pca(lat, lon),
    }

    channels = [
        _interp_nan(_resample_to_n(raw[name]))
        for name in _channel_order
    ]
    return np.column_stack(channels)  # sekil: (500, n_kanal)


def _scale_channels(x):
    """Egitimde kaydedilen MinMaxScaler'lari uygular. Scaler'lar egitimde her
    zaman adimini (500 tanesi) ayri bir 'ozellik' olarak fit edildigi icin,
    burada da her kanali (1, 500) sekliyle veriyoruz — egitimdeki fit sekliyle
    birebir ayni olmali, yoksa scaler yanlis hizalanir."""
    x_scaled = np.empty_like(x)
    for c, scaler in enumerate(_scalers):
        col = x[:, c].reshape(1, -1)
        x_scaled[:, c] = scaler.transform(col).ravel()
    return x_scaled


def lambda_handler(event, context):
    try:
        body = event.get("body", event)
        if isinstance(body, str):
            body = json.loads(body)
        # n8n, node ciktisini disari aktarirken (Download JSON) her zaman bir
        # item listesi olarak sarmalar ([{...}]) — n8n'in kendi ic veri formati
        # boyle. Tek elemanli bir liste gelirse iceki dict'i aliyoruz.
        if isinstance(body, list):
            if len(body) != 1:
                raise ValueError(f"Tek bir aktivite bekleniyordu, {len(body)} item geldi")
            body = body[0]

        hr_p10, hr_p95 = _hr_percentiles(body.get("heartrate", {}).get("data", []))

        x = _extract_channels(body)
        x_scaled = _scale_channels(x)
        x_batch = x_scaled[np.newaxis, ...]  # (1, 500, n_kanal)

        proba = float(_model.predict(x_batch, verbose=0)[0, 0])
        zone = "anaerobik" if proba >= 0.5 else "aerobik"
        confidence = abs(proba - 0.5) * 2

        return {
            "statusCode": 200,
            "body": json.dumps({
                "training_zone": zone,
                "anaerobic_probability": round(proba, 4),
                "confidence": round(confidence, 4),
                "hr_p10": round(hr_p10, 1) if hr_p10 is not None else None,
                "hr_p95": round(hr_p95, 1) if hr_p95 is not None else None,
            }),
        }
    except Exception as e:
        return {"statusCode": 400, "body": json.dumps({"error": str(e)})}


if __name__ == "__main__":
    # Lokal hizli test — AWS'ye deploy etmeden once calisip calismadigini gormek icin.
    # sample_streams.json, Strava streams API cevabinin bir ornegini icermeli.
    with open("sample_streams.json") as f:
        sample = json.load(f)
    print(lambda_handler({"body": json.dumps(sample)}, None))
