# -*- coding: utf-8 -*-
"""
EĞİTİM — TF-IDF + lojistik regresyon (işlemcide birkaç dakika).  python egit/veri.py && python egit/egit.py

C (düzenlileştirme) doğrulama doğruluğuna göre seçilir; sınava burada bakılmaz.
Çıktı model/uslup.npz: üsluplar, sözlük, idf, ağırlık, sapma (çalışma anı yalnız numpy).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK))
from egit.veri import USLUPLAR  # noqa: E402
from turkce_uslup.model import ozellikler  # noqa: E402


def oku(ad: str) -> tuple[list[str], np.ndarray, list[str]]:
    x, y, k = [], [], []
    for s in (KOK / "veri" / f"{ad}.tsv").read_text(encoding="utf-8").splitlines():
        c, u, kaynak = s.split("\t")
        x.append(c)
        y.append(USLUPLAR.index(u))
        k.append(kaynak)
    return x, np.array(y), k


def main() -> int:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    xe, ye, _ = oku("egitim")
    xd, yd, _ = oku("dogrulama")
    v = TfidfVectorizer(analyzer=ozellikler, sublinear_tf=True, min_df=3, max_features=150_000)
    Xe, Xd = v.fit_transform(xe), v.transform(xd)
    en_iyi = None
    for C in (1.0, 3.0, 10.0, 30.0):
        lr = LogisticRegression(C=C, max_iter=3000).fit(Xe, ye)
        d = float((lr.predict(Xd) == yd).mean())
        print(f"C={C:5.1f}  doğrulama doğruluğu %{100 * d:.1f}", flush=True)
        if en_iyi is None or d > en_iyi[0]:
            en_iyi = (d, C, lr)
    d, C, lr = en_iyi
    sozluk = v.get_feature_names_out()
    hedef = KOK / "model"
    hedef.mkdir(exist_ok=True)
    np.savez_compressed(hedef / "uslup.npz", usluplar=np.array(USLUPLAR), sozluk=np.array(sozluk, dtype=str),
                        idf=v.idf_.astype(np.float32), agirlik=lr.coef_.astype(np.float32),
                        sapma=lr.intercept_.astype(np.float32))
    print(f"seçilen C={C} (doğrulama %{100 * d:.1f}); {len(sozluk)} özellik → {hedef / 'uslup.npz'} "
          f"({(hedef / 'uslup.npz').stat().st_size / 1e6:.1f} MB)")
    # numpy çalışma anı sklearn ile aynı mı?
    from turkce_uslup.model import UslupModeli
    m = UslupModeli(hedef)
    ayni = float((m.olasiliklar(xd).argmax(1) == lr.predict(Xd)).mean())
    sapma = float(np.abs(m.olasiliklar(xd) - lr.predict_proba(Xd)).max())
    print(f"numpy çalışma anı ↔ sklearn: aynı tahmin %{100 * ayni:.2f}, en büyük olasılık farkı {sapma:.2e}")
    return 0 if ayni == 1.0 else 1


if __name__ == "__main__":
    sys.exit(main())
