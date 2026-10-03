# turkce-uslup

Türkçe metnin **üslubunu** bulur ve seslendirme için **okuma ayarı** önerir: hız, tonlama genişliği,
cümle arası duraklama, virgül duraklaması. Bağımlılığı yalnız numpy, model 1,3 MB, yanıtı anlıktır.

```python
from turkce_uslup import uslup, okuma_ayari

r = uslup("Bir varmış bir yokmuş, evvel zaman içinde küçük bir köyde yaşlı bir değirmenci yaşarmış.")
r.uslup      # 'masal'
r.ayar       # Okuma(hiz=0.88, tonlama=1.3, duraklama_ms=700, virgul_ms=100)

okuma_ayari("haber", hiz=1.2)     # stüdyoda değiştirilen ayar: üslubun geri kalanı korunur
```

## Neden

Aynı nötr cümle üsluba göre başka okunur. Haber tempolu ve düz okunur. Masal yavaş, hareketli ve
duraklamalı okunur. Sohbet rahat ve hızlı akar. Duygu (neşeli, üzgün…) cümlenin tonudur; üslup ise
metnin bütününün türüdür. İkisi birbirinden ayrı ayarlanmalıdır. Duygu için kardeş proje
[turkce-duygu](../turkce-duygu)'ya bakın.

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
   "sayın yolcularımız", "değerli müşterilerimiz", "dikkatinize sunarız".
2. **Model.** TF-IDF özellikleri (karakter 2-5 gram, sözcük, noktalama) ve çok sınıflı lojistik regresyon.
   Metnin üslubu, cümlelerin log-olasılıklarının ortalamasıdır. Çalışma anında yalnız numpy kullanılır.
   sklearn çıktısıyla birebir aynıdır (doğrulamada %100 aynı tahmin, en büyük olasılık farkı 4·10⁻⁷).

| Üslup | Eğitim verisi | Lisans |
|---|---|---|
| haber, talimat, deneme, bilgi (bir kısmı) | [UD Turkish-BOUN](https://github.com/UniversalDependencies/UD_Turkish-BOUN) (Türkçe Ulusal Derlem'den; tür cümle kimliğinde) | CC BY-SA 4.0 |
| bilgi | Türkçe Vikipedi | CC BY-SA 4.0 |
| anlati | Common Voice Türkçe cümle derlemi | CC0 |
| sohbet | [XED](https://github.com/Helsinki-NLP/XED) Türkçe altyazılar | CC BY 4.0 |

## Doğruluk

Sınav kümesinde ölçüldü (1.604 cümle; eğitimde ve ayar seçiminde kullanılmadı).

| Ölçü | Doğruluk |
|---|---|
| Tek cümle (6 üslup, şans %17) | %52,5 |
| Metin, 3 cümle | %71,2 |
| Metin, 5 cümle | %77,0 |
| Metin, 10 cümle | **%90,6** |

"Metin": aynı üsluptan rastgele seçilen k cümle. Veride gerçek paragraf yoktur.

**Sınırlar:**
- **Kaynak etkisi.** "bilgi" üslubunda Vikipedi cümleleri %89 doğru bulunur, BOUN biyografileri yalnız %23.
  Model türün yanında kaynağın yazım tarzını da öğreniyor.
- **Benzer türler karışır.** Gazete haberi, deneme ve biyografi birbirine benzer yazı dilleridir.
- **Tek cümle güvenilir değildir.** Kısa metinlerde stüdyoda elle seçim önerilir.
- **Kurallar yalnız açık kalıpları tanır.** Tekerlemesiz bir masal model tarafından "anlatı" olarak
  bulunur.

## Yeniden eğitmek

```
pip install turkce-uslup[egitim]
python egit/veri.py         # kaynaklar sabit sürümlerden; MD5'e göre bölme
python egit/egit.py         # birkaç dakika, işlemcide
python degerlendirme/olc.py
```

## Lisans

Kod MIT ([LICENSE](LICENSE)). Model (`model/uslup.npz`) UD Turkish-BOUN ve Vikipedi türevi olduğu için
**CC BY-SA 4.0**; XED (CC BY 4.0) ve Common Voice (CC0) atıfları yukarıda.

Kardeş projeler: [turkce-duygu](../turkce-duygu), [turkce-vurgu](../turkce-vurgu),
[turkce-normalize](../turkce-normalize), [turkce-plbert](../turkce-plbert), [turkce-tts-sinav](../turkce-tts-sinav).
