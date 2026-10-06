# -*- coding: utf-8 -*-
"""
BERTurk DENEMESİ — tek cümlede TF-IDF modelinden (sınavda %52,5) iyi olur mu?

dbmdz/bert-base-turkish-cased, 7 üslup, aynı eğitim/doğrulama/sınav bölmesi. En iyi tur doğrulama
doğruluğuna göre; sınava yalnız sonda bakılır. Ölçüler degerlendirme/olc.py ile aynı (tek cümle,
3/5/10 cümlelik metin, kaynak kaynak). Ayrıca TF-IDF + BERTurk birleşimi (log-olasılık ortalaması).
GPU varsa kullanır (RTX 3050'de birkaç dakika); yoksa işlemcide saatler sürer.

    python egit/egit_bert.py            → model/bert/ (Hugging Face biçimi) + degerlendirme/bert_sonuc.json
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK))
from egit.egit import oku  # noqa: E402
from egit.veri import USLUPLAR  # noqa: E402
from turkce_uslup.model import UslupModeli  # noqa: E402

TABAN = "dbmdz/bert-base-turkish-cased"
UZUNLUK = 96


def olc(lp: np.ndarray, ys: np.ndarray, ks: list[str]) -> dict:
    tahmin = lp.argmax(1)
    s = {"cumle": round(float((tahmin == ys).mean()), 4)}
    rnd = np.random.default_rng(0)
    for k in (3, 5, 10):
        dogru = toplam = 0
        for i in range(len(USLUPLAR)):
            idx = np.flatnonzero(ys == i)
            for _ in range(200):
                sec = rnd.choice(idx, size=k, replace=False)
                dogru += int(lp[sec].mean(0).argmax() == i)
                toplam += 1
        s[f"metin_{k}"] = round(dogru / toplam, 4)
    for kaynak in sorted(set(ks)):
        idx = [i for i, k in enumerate(ks) if k == kaynak]
        s[f"kaynak_{kaynak}"] = round(float((tahmin[idx] == ys[idx]).mean()), 4)
    return s


@torch.no_grad()
def logp(model, tok, metin: list[str], cihaz: str, parti: int = 64) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(metin), parti):
        g = tok(metin[i:i + parti], padding=True, truncation=True, max_length=UZUNLUK, return_tensors="pt").to(cihaz)
        with torch.autocast(cihaz, dtype=torch.float16, enabled=cihaz == "cuda"):
            z = model(**g).logits.float()
        out.append(torch.log_softmax(z, -1).cpu().numpy())
    return np.concatenate(out)


def main() -> int:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    torch.manual_seed(0)
    random.seed(0)
    cihaz = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"cihaz: {cihaz}" + (f" ({torch.cuda.get_device_name()})" if cihaz == "cuda" else ""), flush=True)
    xe, ye, _ = oku("egitim")
    xd, yd, _ = oku("dogrulama")
    xs, ys, ks = oku("sinav")
    tok = AutoTokenizer.from_pretrained(TABAN)
    model = AutoModelForSequenceClassification.from_pretrained(
        TABAN, num_labels=len(USLUPLAR), id2label=dict(enumerate(USLUPLAR)),
        label2id={u: i for i, u in enumerate(USLUPLAR)}).to(cihaz)
    parti, tur = 32, 4
    adim_tur = math.ceil(len(xe) / parti)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    plan = get_linear_schedule_with_warmup(opt, int(0.06 * adim_tur * tur), adim_tur * tur)
    olcek = torch.amp.GradScaler(enabled=cihaz == "cuda")
    hedef = KOK / "model" / "bert"
    en_iyi = -1.0
    sira = list(range(len(xe)))
    for t in range(tur):
        model.train()
        random.shuffle(sira)
        t0 = time.time()
        for b in range(adim_tur):
            idx = sira[b * parti:(b + 1) * parti]
            g = tok([xe[i] for i in idx], padding=True, truncation=True, max_length=UZUNLUK, return_tensors="pt").to(cihaz)
            y = torch.tensor(ye[idx], dtype=torch.long, device=cihaz)
            with torch.autocast(cihaz, dtype=torch.float16, enabled=cihaz == "cuda"):
                kayip = torch.nn.functional.cross_entropy(model(**g).logits.float(), y)
            olcek.scale(kayip).backward()
            olcek.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            olcek.step(opt)
            olcek.update()
            plan.step()
            opt.zero_grad()
            if b % 100 == 0:
                print(f"tur {t} adım {b}/{adim_tur} kayıp {kayip.item():.3f} ({(time.time() - t0) / (b + 1):.2f} sn/adım)",
                      flush=True)
        d = float((logp(model, tok, xd, cihaz).argmax(1) == yd).mean())
        print(f"== tur {t}: doğrulama %{100 * d:.1f}", flush=True)
        if d > en_iyi:
            en_iyi = d
            model.save_pretrained(hedef)
            tok.save_pretrained(hedef)
            print("   kaydedildi", flush=True)

    model = AutoModelForSequenceClassification.from_pretrained(hedef).to(cihaz)
    lb = logp(model, tok, xs, cihaz)
    lt = np.log(UslupModeli(KOK / "model").olasiliklar(xs) + 1e-9)
    sonuc = {"dogrulama": round(en_iyi, 4), "tfidf": olc(lt, ys, ks), "bert": olc(lb, ys, ks),
             "birlesim": olc((lt + lb) / 2, ys, ks)}
    print("\nSINAV (" + ", ".join(f"{k}" for k in sonuc["bert"]) + ")")
    for ad in ("tfidf", "bert", "birlesim"):
        print(f"  {ad:9s} " + " · ".join(f"{k} %{100 * v:.1f}" for k, v in sonuc[ad].items()))
    (KOK / "degerlendirme" / "bert_sonuc.json").write_text(json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
