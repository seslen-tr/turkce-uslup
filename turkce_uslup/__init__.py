# -*- coding: utf-8 -*-
"""
TÜRKÇE ÜSLUP — Türkçe metnin türünü (haber, bilgi, talimat/duyuru, deneme, anlatı, masal, sohbet) bulur
ve seslendirme için okuma ayarı önerir: hız, tonlama genişliği, cümle arası duraklama.

    from turkce_uslup import uslup, okuma_ayari
    r = uslup("Bir varmış bir yokmuş, evvel zaman içinde bir padişah varmış...")
    r.uslup, r.ayar        # 'masal', Okuma(hiz=0.88, tonlama=1.3, duraklama_ms=700, ...)
    okuma_ayari("haber", hiz=1.1)      # stüdyoda değiştirilen ayar

Üslup metnin bütününün özelliğidir: tek cümlede güvenilir değildir, birkaç cümleden sonra güçlenir.
Model yoksa yalnız kurallar (masal tekerlemeleri, duyuru kalıpları) çalışır, gerisi "bilgi" sayılır.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, replace

from .kurallar import kural_uslubu

__all__ = ["uslup", "okuma_ayari", "Okuma", "Sonuc", "USLUPLAR", "OKUMA"]
__version__ = "0.1.0"

USLUPLAR = ("haber", "bilgi", "talimat", "deneme", "anlati", "masal", "sohbet")
AD = {"haber": "haber", "bilgi": "bilgi / ansiklopedi", "talimat": "talimat / duyuru", "deneme": "deneme / köşe yazısı",
      "anlati": "hikâye / anlatı", "masal": "masal", "sohbet": "sohbet / diyalog"}


@dataclass(frozen=True)
class Okuma:
    """Seslendirme ayarı. hiz: 1 = modelin doğal hızı; tonlama: perde dalgalanmasının çarpanı (1 = doğal,
    >1 daha hareketli); duraklama_ms: cümleler arası sessizlik; virgul_ms: virgülde ek sessizlik."""
    hiz: float = 1.0
    tonlama: float = 1.0
    duraklama_ms: int = 400
    virgul_ms: int = 0


# Başlangıç değerleri: genel okuma pratiği (haber tempolu ve düz, masal yavaş ve hareketli, sohbet rahat
# ve hızlı). Ölçülmüş Türkçe değerler değildir; kulakla ayarlanır ve stüdyoda değiştirilebilir.
OKUMA = {
    "haber": Okuma(hiz=1.05, tonlama=0.9, duraklama_ms=350),
    "bilgi": Okuma(hiz=1.0, tonlama=1.0, duraklama_ms=400),
    "talimat": Okuma(hiz=0.95, tonlama=0.85, duraklama_ms=500, virgul_ms=80),
    "deneme": Okuma(hiz=0.97, tonlama=1.0, duraklama_ms=450),
    "anlati": Okuma(hiz=0.93, tonlama=1.15, duraklama_ms=550, virgul_ms=60),
    "masal": Okuma(hiz=0.88, tonlama=1.3, duraklama_ms=700, virgul_ms=100),
    "sohbet": Okuma(hiz=1.07, tonlama=1.1, duraklama_ms=250),
}


def okuma_ayari(ad: str, **degisiklik) -> Okuma:
    """Üslubun varsayılan ayarı, istenen alanlar değiştirilmiş olarak (stüdyo kaydırıcıları)."""
    return replace(OKUMA[ad], **degisiklik)


@dataclass
class Sonuc:
    uslup: str
    ayar: Okuma
    gerekce: str
    olasiliklar: dict[str, float] = field(default_factory=dict)

    def sozluk(self) -> dict:
        return {"uslup": self.uslup, "ad": AD[self.uslup], "ayar": asdict(self.ayar), "gerekce": self.gerekce,
                "olasiliklar": self.olasiliklar}


def _cumleler(metin: str) -> list[str]:
    try:
        from turkce_normalize import cumleleri_ayir
        return [c for c in cumleleri_ayir(metin, 0) if c.strip()]
    except ImportError:
        return [c for c in re.split(r"(?<=[.!?…])\s+", metin.strip()) if c]


def uslup(metin: str, model=None) -> Sonuc:
    """Metnin üslubu ve okuma ayarı. Önce kurallar (kaynaklı kalıplar), sonra model (varsa)."""
    k = kural_uslubu(metin)
    if k:
        ad, neden = k
        return Sonuc(ad, OKUMA[ad], f"kural: {neden}")
    if model is None:
        try:
            from .model import varsayilan
            model = varsayilan()
        except FileNotFoundError:
            return Sonuc("bilgi", OKUMA["bilgi"], "model yok, varsayılan")
    cumleler = _cumleler(metin) or [metin]
    p = model.metin_olasiligi(cumleler)
    olas = {u: round(float(x), 4) for u, x in zip(model.usluplar, p)}
    ad = max(olas, key=olas.get)
    return Sonuc(ad, OKUMA[ad], f"model ({len(cumleler)} cümle, %{100 * olas[ad]:.0f})", olas)
