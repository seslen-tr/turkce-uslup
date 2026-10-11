# Değişiklikler

## 0.1.0 — ilk açık sürüm

- İlk yayın. Ayrıntılı ölçümler README'de.
- 7 üslup (haber, bilgi, talimat/duyuru, deneme, anlatı, masal, sohbet) ve her biri için okuma ayarı
  (`Okuma`: hız, tonlama, cümle arası ve virgül duraklaması); `okuma_ayari(ad, **degisiklik)`.
- Kurallar: masal tekerlemeleri, sesli duyuru kalıpları, yönerge dizisi (kibar emir kipi).
- TF-IDF + lojistik regresyon modeli pakette (yalnız numpy, internetsiz).
- İsteğe bağlı BERTurk modeli (ONNX, fp32, `[bert]` eki); yerelde yoksa Hugging Face'ten
  (`seslen-tr/turkce-uslup`) iner, bulunamazsa TF-IDF'e düşülür.
