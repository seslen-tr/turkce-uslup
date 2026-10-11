# turkce-uslup

> **English:** Detects the genre of a Turkish text (news, informative, instructions/announcement, essay, narrative, fairy tale, conversation) and suggests reading settings for TTS: speed, pitch range, pause between sentences and after commas. Ships with a 1.3 MB TF-IDF model (numpy only, works offline); an optional fine-tuned BERTurk model is downloaded from Hugging Face.

Türkçe metnin **üslubunu** bulur ve seslendirme için **okuma ayarı** önerir: hız, tonlama genişliği,
cümle arası duraklama, virgül duraklaması. Temel kurulumun bağımlılığı yalnız numpy: pakette gelen
1,3 MB'lık TF-IDF modeli internetsiz ve anlık çalışır. İsteğe bağlı BERTurk modeli daha doğrudur.

```python
from turkce_uslup import uslup, okuma_ayari

r = uslup("Bir varmış bir yokmuş, evvel zaman içinde küçük bir köyde yaşlı bir değirmenci yaşarmış.")
r.uslup      # 'masal'
r.ayar       # Okuma(hiz=0.88, tonlama=1.3, duraklama_ms=700, virgul_ms=100)

okuma_ayari("haber", hiz=1.2)     # stüdyoda değiştirilen ayar: üslubun geri kalanı korunur
```

## Kurulum

```
pip install turkce-uslup          # TF-IDF modeli pakette; yalnız numpy
pip install "turkce-uslup[bert]"  # BERTurk (onnxruntime, tokenizers); model ilk kullanımda Hugging Face'ten iner
```

- BERTurk modeli şu sırayla aranır: `BertUslupModeli(klasor)`, `TURKCE_USLUP_BERT` ortam değişkeni,
  paketin ya da depo kökünün `model/bert_onnx/` klasörü, en son Hugging Face deposu (varsayılan
  `seslen-tr/turkce-uslup`, `TURKCE_USLUP_HF` ile değişir; `TURKCE_USLUP_INDIRME=0` ya da
  `HF_HUB_OFFLINE=1` indirmeyi kapatır). Bulunamazsa TF-IDF'e düşülür.
- TF-IDF modeli (`uslup.npz`) pakette gelir; başka bir klasör `TURKCE_USLUP_MODEL` ile gösterilir.
- `TURKCE_USLUP_MOTOR=tfidf` BERTurk kurulu olsa da TF-IDF kullandırır.

## API

- `uslup(metin, model=None)` → `Sonuc`: `uslup` (üslup adı), `ayar` (`Okuma`), `gerekce` ("kural: …" ya da
  "model (n cümle, %…)"), `olasiliklar` (model kararıysa üslup başına olasılık; kural kararında boş).
  `Sonuc.sozluk()` JSON'a uygun sözlük verir (Türkçe görünen ad dahil).
- `okuma_ayari(ad, **degisiklik)` → üslubun varsayılan `Okuma` ayarı, verilen alanlar değiştirilmiş.
- `Okuma(hiz, tonlama, duraklama_ms, virgul_ms)`; `OKUMA` (üslup → varsayılan ayar), `USLUPLAR` (7 üslup).
- `model` verilmezse `turkce_uslup.model.varsayilan()` kullanılır. Metin cümlelere bölünür:
  [turkce-normalize](https://github.com/seslen-tr/turkce-normalize) kuruluysa onun bölücüsüyle, yoksa
  `. ! ? …` ardından boşlukta.

## Neden

Aynı nötr cümle üsluba göre başka okunur. Haber tempolu ve düz okunur. Masal yavaş, hareketli ve
duraklamalı okunur. Sohbet rahat ve hızlı akar. Duygu (neşeli, üzgün…) cümlenin tonudur; üslup ise
metnin bütününün türüdür. İkisi birbirinden ayrı ayarlanmalıdır. Duygu için kardeş proje
[turkce-duygu](https://github.com/seslen-tr/turkce-duygu)'ya bakın.

## Üsluplar ve okuma ayarları

| Üslup | Örnek | hız | tonlama | duraklama | virgül |
|---|---|---|---|---|---|
| haber | ulusal gazete haberi | 1,05 | 0,9 | 350 ms | — |
| bilgi | ansiklopedi, biyografi | 1,0 | 1,0 | 400 ms | — |
| talimat | kullanım kılavuzu, sesli duyuru | 0,95 | 0,85 | 500 ms | +80 ms |
| deneme | deneme, köşe yazısı, popüler kültür | 0,97 | 1,0 | 450 ms | — |
| anlati | hikâye, roman | 0,93 | 1,15 | 550 ms | +60 ms |
| masal | masal (tekerlemeyle tanınır) | 0,88 | 1,3 | 700 ms | +100 ms |
| sohbet | diyalog, konuşma dili | 1,07 | 1,1 | 250 ms | — |

**Ayarların anlamı:**
- **hız:** 1 = sesin doğal hızı.
- **tonlama:** perdenin ortalama etrafındaki dalgalanmasının çarpanı. 1 doğal, büyükse daha hareketli.
- **duraklama:** cümleler arası sessizlik.
- **virgül:** virgülde eklenen sessizlik.

Değerler genel okuma pratiğine dayanan **başlangıç değerleridir**, ölçülmüş Türkçe normlar değildir.
Kulakla ayarlanır ve her biri `okuma_ayari(..., alan=değer)` ile değiştirilebilir.

## Nasıl çalışır

1. **Kurallar.** Masal tekerlemeleri: "bir varmış bir yokmuş" (TDK, "bir masala başlarken … söylenen bir
   tekerleme"), "evvel zaman içinde", "kalbur saman içinde", "onlar ermiş muradına". Duyuru kalıpları:
   "sayın yolcularımız", "değerli müşterilerimiz", "dikkatinize sunarız". Yönerge dizileri: tarif,
   kılavuz ve yönergeler kibar emir kipiyle yazılır (-(y)In, -(y)InIz; Göksel & Kerslake §21.4.4.2).
   En az iki cümle bu kiple bitiyorsa ya da bir cümle sıra sözüyle ("önce", "ardından", "son olarak")
   başlayıp bu kiple bitiyorsa talimat sayılır. Kişi ekleri (yanılıyor-sunuz, yap-tınız) ve konuşmadaki
   ünlemli buyruklar ("Bekleyin!") hariç tutulur. Bu kural doğrulama ve sınav kümelerinde doğruluğu
   değiştirmedi; modelin talimatta zayıf kaldığı gerçek yönergeleri yakalar.
2. **Model.** İki model var; `turkce_uslup.model.varsayilan()` BERTurk varsa onu, yoksa TF-IDF'i kullanır
   (`TURKCE_USLUP_MOTOR=tfidf` ile TF-IDF'e zorlanır). Metnin üslubu, cümlelerin log-olasılıklarının
   ortalamasıdır.
   - **BERTurk** ([dbmdz/bert-base-turkish-cased](https://huggingface.co/dbmdz/bert-base-turkish-cased))
     ince ayarı, ONNX olarak (`bert_onnx/`, 443 MB, cümle başına ~10–25 ms). ONNX ↔ PyTorch en büyük
     olasılık farkı 5·10⁻⁶. int8 sıkıştırma doğrulamada 0,5 puan kaybettirdiği için fp32. Gerekenler:
     onnxruntime, tokenizers (`[bert]` eki). Yalnız 6 üslubu ayırır; masal kuralla bulunur.
   - **TF-IDF** (karakter 2-5 gram, sözcük, noktalama) + çok sınıflı lojistik regresyon (`turkce_uslup/model/uslup.npz`).
     Çalışma anında yalnız numpy; sklearn çıktısıyla birebir aynı (en büyük olasılık farkı 4·10⁻⁷).
     Hafif yedek.

| Üslup | Eğitim verisi | Lisans |
|---|---|---|
| haber, talimat, deneme, bilgi (bir kısmı) | [UD Turkish-BOUN](https://github.com/UniversalDependencies/UD_Turkish-BOUN) (Türkçe Ulusal Derlem'den; tür cümle kimliğinde) | CC BY-SA 4.0 |
| bilgi | Türkçe Vikipedi | CC BY-SA 4.0 |
| anlati | Common Voice Türkçe cümle derlemi | CC0 |
| sohbet | [XED](https://github.com/Helsinki-NLP/XED) Türkçe altyazılar | CC BY 4.0 |

## Doğruluk

Sınav kümesinde ölçüldü (1.604 cümle; eğitimde ve ayar seçiminde kullanılmadı).

| Ölçü | BERTurk | TF-IDF |
|---|---|---|
| Tek cümle (6 üslup, şans %17) | **%68,5** | %52,5 |
| Metin, 3 cümle | **%89,9** | %71,2 |
| Metin, 5 cümle | **%96,1** | %77,0 |
| Metin, 10 cümle | **%99,7** | %90,6 |

İkisinin birleşimi (log-olasılık ortalaması) BERTurk'ten anlamlı ölçüde iyi değil (tek cümle %68,6).
Kaynak kaynak sonuçlar `degerlendirme/bert_sonuc.json` dosyasında.

"Metin": aynı üsluptan rastgele seçilen k cümle. Veride gerçek paragraf yoktur.

**Sınırlar:**
- **Kaynak etkisi.** "bilgi" üslubunda Vikipedi cümleleri %99 (TF-IDF %89) doğru bulunur, BOUN
  biyografileri yalnız %37 (TF-IDF %23). Model türün yanında kaynağın yazım tarzını da öğreniyor.
- **Talimat verisi tek kaynaklı.** Eğitimdeki talimat cümleleri BOUN'un "ins" türünden; gündelik tarif ve
  kullanım yönergelerini model çoğu zaman "deneme" sanıyor. Bu yüzden kural eklendi (yukarıda).
- **Benzer türler karışır.** Gazete haberi, deneme ve biyografi birbirine benzer yazı dilleridir.
- **Tek cümle güvenilir değildir.** Kısa metinlerde stüdyoda elle seçim önerilir.
- **Kurallar yalnız açık kalıpları tanır.** Tekerlemesiz bir masal model tarafından "anlatı" olarak
  bulunur.

## Yeniden eğitmek

```
pip install turkce-uslup[egitim]
python egit/veri.py         # kaynaklar sabit sürümlerden; MD5'e göre bölme → veri/{egitim,dogrulama,sinav}.tsv
python egit/egit.py         # TF-IDF: birkaç dakika, işlemcide → model/uslup.npz
python degerlendirme/olc.py # → degerlendirme/sonuc.json
python egit/egit_bert.py    # BERTurk: GPU'da birkaç dakika (RTX 3050'de ~6 dk), işlemcide saatler
                            # → model/bert + degerlendirme/bert_sonuc.json
python egit/aktar_bert.py   # ONNX + eşdeğerlik kapısı → model/bert_onnx
```

- `[egitim]` eki yalnız TF-IDF için yeter; BERTurk eğitimi ve aktarımı ayrıca `torch`, `transformers`,
  `onnx` ve `onnxruntime` ister.
- `egit/egit.py` depo kökündeki `model/uslup.npz`'yi yazar; pakette gelen kopya
  `turkce_uslup/model/uslup.npz`'dir, yeni modeli paketlemek için oraya kopyalanır.
- BERTurk modeli Hugging Face'e `python hf/yukle.py` ile yüklenir (varsayılan gizli depo, `--acik` ile
  herkese açık; model kartı `hf/README.md`).

## Düzen

| Yol | İçerik |
|---|---|
| `turkce_uslup/__init__.py` | `uslup`, `okuma_ayari`, `Okuma`, `OKUMA`, `Sonuc` |
| `turkce_uslup/kurallar.py` | masal tekerlemeleri, duyuru kalıpları, yönerge dizisi kuralı |
| `turkce_uslup/model.py` | TF-IDF (`UslupModeli`) ve BERTurk ONNX (`BertUslupModeli`) modelleri, `varsayilan()` |
| `turkce_uslup/model/uslup.npz` | pakette gelen TF-IDF modeli |
| `egit/` | veri hazırlama, TF-IDF ve BERTurk eğitimi, ONNX aktarımı |
| `degerlendirme/` | `olc.py`, `sonuc.json` (TF-IDF), `bert_sonuc.json` (BERTurk, TF-IDF, birleşim) |
| `hf/` | Hugging Face model kartı ve yükleme betiği |
| `tests/` | kural, okuma ayarı ve model testleri (pakette gelen TF-IDF modeliyle; internet gerekmez) |

## Lisans

Kod MIT ([LICENSE](LICENSE)). Modeller (`uslup.npz`, `bert_onnx/`) UD Turkish-BOUN ve Vikipedi
türevi olduğu için **CC BY-SA 4.0**; BERTurk başlangıç modeli MIT; XED (CC BY 4.0) ve Common Voice (CC0) atıfları yukarıda.

Kardeş projeler: [turkce-duygu](https://github.com/seslen-tr/turkce-duygu), [turkce-vurgu](https://github.com/seslen-tr/turkce-vurgu),
[turkce-normalize](https://github.com/seslen-tr/turkce-normalize), [turkce-plbert](https://github.com/seslen-tr/turkce-plbert), [turkce-tts-sinav](https://github.com/seslen-tr/turkce-tts-sinav).
