# -*- coding: utf-8 -*-
"""
BERTurk üslup modelini ONNX'e aktar: model/bert (egit/egit_bert.py) → model/bert_onnx/
  uslup.onnx, tokenizer.json, etiketler.json

Kapılar: (1) ONNX ↔ PyTorch olasılık farkı ≤ 1e-3; (2) int8 (dinamik nicemleme) doğrulama doğruluğunu
0,5 puandan fazla düşürmezse int8 (~4× küçük), yoksa fp32 yazılır. Sınava burada bakılmaz.

    python egit/aktar_bert.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK))
from egit.egit import oku  # noqa: E402

KAYNAK, HEDEF = KOK / "model" / "bert", KOK / "model" / "bert_onnx"
ADLAR = ["input_ids", "attention_mask", "token_type_ids"]
UZUNLUK = 96


def olasilik(oturum, tok, metin: list[str], parti: int = 64) -> np.ndarray:
    out = []
    for i in range(0, len(metin), parti):
        g = tok(metin[i:i + parti], padding=True, truncation=True, max_length=UZUNLUK, return_tensors="np")
        z = oturum.run(None, {k: g[k].astype(np.int64) for k in ADLAR})[0]
        z = np.exp(z - z.max(1, keepdims=True))
        out.append(z / z.sum(1, keepdims=True))
    return np.concatenate(out)


def main() -> int:
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(KAYNAK)
    model = AutoModelForSequenceClassification.from_pretrained(KAYNAK).eval()
    gecici = KOK / "model" / "bert_onnx_gecici"
    gecici.mkdir(parents=True, exist_ok=True)
    fp32, int8 = gecici / "uslup_fp32.onnx", gecici / "uslup_int8.onnx"
    ornek = tok(["Bu bir deneme cümlesidir."], return_tensors="pt")

    class Sar(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.m(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits

    xd, yd, _ = oku("dogrulama")
    with torch.no_grad():         # başvuru dışa aktarmadan ÖNCE: aktarım modelin dikkat ayarını değiştiriyor
        g = tok(xd[:200], padding=True, truncation=True, max_length=UZUNLUK, return_tensors="pt")
        pt = torch.softmax(model(**g).logits, -1).numpy()
    torch.onnx.export(Sar(model), tuple(ornek[a] for a in ADLAR), str(fp32), input_names=ADLAR, output_names=["logits"],
                      opset_version=17, dynamo=False,
                      dynamic_axes={**{a: {0: "parti", 1: "uzunluk"} for a in ADLAR}, "logits": {0: "parti"}})
    se = ort.SessionOptions()
    o32 = ort.InferenceSession(str(fp32), se, providers=["CPUExecutionProvider"])
    p32 = olasilik(o32, tok, xd)
    sapma = float(np.abs(p32[:200] - pt).max())
    d32 = float((p32.argmax(1) == yd).mean())
    print(f"fp32: ONNX ↔ PyTorch en büyük fark {sapma:.1e}, doğrulama %{100 * d32:.1f}", flush=True)
    if sapma > 1e-3:
        print("KAPI GEÇİLMEDİ (fp32 eşdeğer değil)")
        return 1
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8, per_channel=True)
    p8 = olasilik(ort.InferenceSession(str(int8), se, providers=["CPUExecutionProvider"]), tok, xd)
    d8 = float((p8.argmax(1) == yd).mean())
    print(f"int8: doğrulama %{100 * d8:.1f} ({100 * (d8 - d32):+.1f} puan)", flush=True)
    secilen, ad = (int8, "int8") if d32 - d8 <= 0.005 else (fp32, "fp32")
    HEDEF.mkdir(parents=True, exist_ok=True)
    shutil.copy(secilen, HEDEF / "uslup.onnx")
    shutil.copy(KAYNAK / "tokenizer.json", HEDEF / "tokenizer.json")
    etiket = json.loads((KAYNAK / "config.json").read_text(encoding="utf-8"))["id2label"]
    (HEDEF / "etiketler.json").write_text(json.dumps(
        {"usluplar": [etiket[str(i)] for i in range(len(etiket))], "uzunluk": UZUNLUK, "kesinlik": ad,
         "dogrulama": round(d8 if ad == "int8" else d32, 4)}, ensure_ascii=False, indent=1), encoding="utf-8")
    shutil.rmtree(gecici)
    print(f"yazıldı: {HEDEF} ({ad}, {(HEDEF / 'uslup.onnx').stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
