# -*- coding: utf-8 -*-
"""
ÖLÇÜM — sınav kümesi (eğitimde ve C seçiminde kullanılmadı):
  1) tek cümle doğruluğu ve karışıklık tablosu
  2) metin düzeyi: aynı üsluptan rastgele k cümle bir "metin" sayılır (k = 3, 5, 10), metnin üslubu
     cümlelerin log-olasılık ortalamasından. Gerçek paragraf değildir (BOUN cümleleri tek tek örneklenmiş).
  3) kaynaktan kaynağa genelleme: bilgi sınıfı hem BOUN biyografi hem Vikipedi'den gelir; her biri
     ayrı ölçülür (model kaynağın kendisini mi öğrendi, türü mü?).

    python degerlendirme/olc.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK))
from egit.egit import oku  # noqa: E402
from egit.veri import USLUPLAR  # noqa: E402
from turkce_uslup.model import UslupModeli  # noqa: E402


def main() -> int:
    m = UslupModeli(KOK / "model")
    xs, ys, ks = oku("sinav")
    p = m.olasiliklar(xs)
    tahmin = p.argmax(1)
    sonuc = {"cumle_dogruluk": round(float((tahmin == ys).mean()), 4)}
    print(f"TEK CÜMLE: %{100 * sonuc['cumle_dogruluk']:.1f} ({len(xs)} cümle)")
    print("karışıklık (satır: doğru, sütun: tahmin)\n" + " " * 9 + " ".join(f"{u[:7]:>7s}" for u in USLUPLAR))
    for i, u in enumerate(USLUPLAR):
        satir = [(tahmin[ys == i] == j).sum() for j in range(len(USLUPLAR))]
        print(f"{u:8s} " + " ".join(f"{n:7d}" for n in satir) + f"   %{100 * satir[i] / max(1, sum(satir)):.0f}")
    lp = np.log(p + 1e-9)
    rnd = np.random.default_rng(0)
    for k in (3, 5, 10):
        dogru = toplam = 0
        for i in range(len(USLUPLAR)):
            idx = np.flatnonzero(ys == i)
            for _ in range(200):
                sec = rnd.choice(idx, size=k, replace=False)
                dogru += int(lp[sec].mean(0).argmax() == i)
                toplam += 1
        sonuc[f"metin_{k}_cumle"] = round(dogru / toplam, 4)
        print(f"METİN ({k} cümle): %{100 * dogru / toplam:.1f}")
    for kaynak in sorted(set(ks)):
        idx = [i for i, k in enumerate(ks) if k == kaynak]
        d = float((tahmin[idx] == ys[idx]).mean())
        sonuc[f"kaynak_{kaynak}"] = round(d, 4)
        print(f"  kaynak {kaynak:12s} {len(idx):4d} cümle, doğru %{100 * d:.1f}")
    (KOK / "degerlendirme" / "sonuc.json").write_text(json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
