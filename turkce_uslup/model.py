# -*- coding: utf-8 -*-
"""
ÜSLUP MODELLERİ
  BertUslupModeli  BERTurk ince ayarı, ONNX (model/bert_onnx; egit/egit_bert.py + egit/aktar_bert.py).
                   Sınavda tek cümle %68,5, 5 cümlelik metin %96,1. Gerekenler: onnxruntime, tokenizers.
  UslupModeli      TF-IDF (karakter 2-5 gram + sözcük + noktalama) + lojistik regresyon; yalnız numpy
                   (model/uslup.npz). Tek cümle %52,5, 5 cümle %77,0. Hafif yedek.
varsayilan(): BERTurk varsa o, yoksa TF-IDF. TURKCE_USLUP_MOTOR=tfidf ile TF-IDF'e zorlanır.
TF-IDF modeli pakette gelir (turkce_uslup/model/uslup.npz). BERTurk yerelde yoksa Hugging Face'ten iner
(TURKCE_USLUP_HF deposu, ilk kullanımda; TURKCE_USLUP_INDIRME=0 ya da HF_HUB_OFFLINE=1 ile kapalı):
pip install "turkce-uslup[bert]".
"""
from __future__ import annotations

import math
import os
import re
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

_PAKET = Path(__file__).resolve().parent
HF_DEPO = os.environ.get("TURKCE_USLUP_HF", "mveyseldelibas/turkce-uslup")


def _hf_indir() -> Path | None:
    """BERTurk üslup modeli yerelde yoksa Hugging Face'ten (bert_onnx/). Kurulu değilse ya da kapalıysa None."""
    if os.environ.get("TURKCE_USLUP_INDIRME") == "0":
        return None
    try:
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(HF_DEPO, allow_patterns=["bert_onnx/*"])) / "bert_onnx"
    except Exception:
        return None


def kucuk(s: str) -> str:
    return unicodedata.normalize("NFC", s.replace("I", "ı").replace("İ", "i")).lower()


def ozellikler(metin: str) -> list[str]:
    """Karakter 2-5 gramları (sözcük sınırlı), sözcükler, noktalama ve biçim ipuçları."""
    k = kucuk(metin)
    out = []
    for w in re.findall(r"[a-zçğıöşüâîû]+", k):
        p = f" {w} "
        out += [p[i:i + n] for n in range(2, 6) for i in range(len(p) - n + 1)]
        out.append("S:" + w)
    out += ["N:" + c for c in re.findall(r"[?!…:;\"“”'()\-–—%]", metin)]
    if re.search(r"\d", metin):
        out.append("B:rakam")
    if metin[:1].isupper():
        out.append("B:buyuk_bas")
    n = len(metin.split())
    out.append("B:uzunluk_" + ("kisa" if n <= 6 else "orta" if n <= 15 else "uzun"))
    return out


def _klasor(klasor: str | Path | None) -> Path:
    for a in ([Path(klasor)] if klasor else []) + ([Path(os.environ["TURKCE_USLUP_MODEL"])]
                                                  if os.environ.get("TURKCE_USLUP_MODEL") else []) \
            + [_PAKET / "model", _PAKET.parent / "model"]:
        if (a / "uslup.npz").exists():
            return a
    raise FileNotFoundError("üslup modeli (uslup.npz) bulunamadı; TURKCE_USLUP_MODEL ile klasörü gösterin")


class UslupModeli:
    def __init__(self, klasor: str | Path | None = None):
        v = np.load(_klasor(klasor) / "uslup.npz", allow_pickle=False)
        self.usluplar: list[str] = [str(x) for x in v["usluplar"]]
        self.sozluk = {str(s): i for i, s in enumerate(v["sozluk"])}
        self.idf, self.w, self.b = v["idf"], v["agirlik"], v["sapma"]

    def logit(self, cumleler: list[str]) -> np.ndarray:
        out = np.zeros((len(cumleler), len(self.usluplar)), dtype=np.float32)
        for r, c in enumerate(cumleler):
            say = Counter(i for i in (self.sozluk.get(f) for f in ozellikler(c)) if i is not None)
            if not say:
                out[r] = self.b
                continue
            idx = np.fromiter(say.keys(), dtype=np.int64)
            deg = np.array([1 + math.log(n) for n in say.values()], dtype=np.float32) * self.idf[idx]
            deg /= np.linalg.norm(deg) or 1.0
            out[r] = self.w[:, idx] @ deg + self.b
        return out

    def olasiliklar(self, cumleler: list[str]) -> np.ndarray:
        z = self.logit(cumleler)
        z = np.exp(z - z.max(1, keepdims=True))
        return z / z.sum(1, keepdims=True)

    def metin_olasiligi(self, cumleler: list[str]) -> np.ndarray:
        """Metnin üslubu: cümlelerin log-olasılık ortalaması (üslup metnin özelliğidir, tek cümle gürültülü)."""
        lp = np.log(self.olasiliklar(cumleler) + 1e-9).mean(0)
        z = np.exp(lp - lp.max())
        return z / z.sum()


class BertUslupModeli:
    def __init__(self, klasor: str | Path | None = None, is_parcacigi: int = 0):
        import json

        import onnxruntime as ort
        from tokenizers import Tokenizer
        k = _bert_klasor(klasor)
        ayar = json.loads((k / "etiketler.json").read_text(encoding="utf-8"))
        self.usluplar: list[str] = ayar["usluplar"]
        self.tok = Tokenizer.from_file(str(k / "tokenizer.json"))
        self.tok.enable_truncation(int(ayar.get("uzunluk", 96)))
        self.tok.enable_padding(pad_id=self.tok.token_to_id("[PAD]") or 0)
        se = ort.SessionOptions()
        if is_parcacigi:
            se.intra_op_num_threads = is_parcacigi
        self.oturum = ort.InferenceSession(str(k / "uslup.onnx"), se, providers=["CPUExecutionProvider"])

    def olasiliklar(self, cumleler: list[str]) -> np.ndarray:
        if not cumleler:
            return np.zeros((0, len(self.usluplar)), dtype=np.float32)
        kod = self.tok.encode_batch(cumleler)
        g = {"input_ids": np.array([e.ids for e in kod], dtype=np.int64),
             "attention_mask": np.array([e.attention_mask for e in kod], dtype=np.int64),
             "token_type_ids": np.array([e.type_ids for e in kod], dtype=np.int64)}
        z = self.oturum.run(None, g)[0]
        z = np.exp(z - z.max(1, keepdims=True))
        return z / z.sum(1, keepdims=True)

    metin_olasiligi = UslupModeli.metin_olasiligi


def _bert_klasor(klasor: str | Path | None) -> Path:
    for a in ([Path(klasor)] if klasor else []) + ([Path(os.environ["TURKCE_USLUP_BERT"])]
                                                  if os.environ.get("TURKCE_USLUP_BERT") else [])             + [_PAKET / "model" / "bert_onnx", _PAKET.parent / "model" / "bert_onnx"]:
        if (a / "uslup.onnx").exists():
            return a
    if (a := _hf_indir()) is not None and (a / "uslup.onnx").exists():
        return a
    raise FileNotFoundError(f"BERTurk üslup modeli (bert_onnx/uslup.onnx) bulunamadı; pip install \"turkce-uslup[bert]\" "
                            f"ile Hugging Face'ten ({HF_DEPO}) iner ya da TURKCE_USLUP_BERT ile klasörü gösterin")


@lru_cache(maxsize=1)
def varsayilan() -> "BertUslupModeli | UslupModeli":
    if os.environ.get("TURKCE_USLUP_MOTOR", "").lower() != "tfidf":
        try:
            return BertUslupModeli()
        except (FileNotFoundError, ImportError):
            pass
    return UslupModeli()
