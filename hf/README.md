---
language: tr
license: cc-by-sa-4.0
library_name: onnx
pipeline_tag: text-classification
base_model: dbmdz/bert-base-turkish-cased
tags: [turkish, genre, register, tts, onnx]
---

# turkce-uslup — Türkçe üslup (metin türü) modeli

> **English:** Fine-tuned BERTurk (ONNX, CPU) that classifies the genre of Turkish sentences (news,
> informative, instructions, essay, narrative, conversation); the turkce-uslup package turns it into reading
> settings for TTS (speed, pitch range, pauses).

İnce ayarlı BERTurk, ONNX (fp32, ~443 MB; int8 doğrulamada 0,5 puan kaybettirdi). Cümle başına ~10–25 ms, işlemcide.
Paket, bu model yoksa kendi içindeki 1,3 MB'lık TF-IDF modeline düşer.

## Kullanım

```python
# pip install "turkce-uslup[bert]"       → model ilk kullanımda buradan iner (bert_onnx/)
from turkce_uslup import uslup
uslup("Fırını 180 dereceye ısıtın. Unu ve şekeri karıştırın. Yumurtaları ekleyin.").uslup   # 'talimat' (yönerge kuralı)
uslup("Hey, nereye gidiyorsun öyle? Dur bir dakika, ben de geliyorum.").olasiliklar         # model kararı
```

Depodaki dosyalar `bert_onnx/` altında: `uslup.onnx` (girdiler `input_ids`, `attention_mask`,
`token_type_ids`; paket 96 parçayla keser), `tokenizer.json`, `etiketler.json` (üslup sırası).

## Ölçüler (ayrılmış sınav, 1.604 cümle, 6 üslup, şans %17)

| | Bu model | TF-IDF |
|---|---|---|
| Tek cümle | **%68,5** | %52,5 |
| Metin, 3 cümle | **%89,9** | %71,2 |
| Metin, 5 cümle | **%96,1** | %77,0 |
| Metin, 10 cümle | **%99,7** | %90,6 |

Masal üslubu kuralla bulunur; talimat için de kural var. Ayrıntı: paketin README'si.

## Veri ve lisans

| Üslup | Veri | Lisans |
|---|---|---|
| haber, talimat, deneme, bilgi (bir kısmı) | UD Turkish-BOUN | CC BY-SA 4.0 |
| bilgi | Türkçe Vikipedi | CC BY-SA 4.0 |
| anlatı | Common Voice Türkçe cümle derlemi | CC0 |
| sohbet | XED Türkçe altyazılar | CC BY 4.0 |

Ağırlıklar **CC BY-SA 4.0** (BOUN ve Vikipedi türevi). Başlangıç: dbmdz/bert-base-turkish-cased (MIT). Kod: MIT.

## Sınırlılıklar

Model türün yanında kaynağın yazım tarzını da öğreniyor: "bilgi" üslubunda Vikipedi cümleleri %99, BOUN
biyografileri %37 doğru bulunur. Tek cümlede karar zayıftır; metin uzadıkça güvenilir olur.
