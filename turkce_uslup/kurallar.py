# -*- coding: utf-8 -*-
"""
KURALLAR — modelin göremediği ya da veride olmayan, kalıplaşmış açılışlar.

  masal    TDK: "bir varmış bir yokmuş" — "bir masala başlarken, 'eskiden' anlamında söylenen bir
           tekerleme". Aynı geleneğin öteki başlangıç ve bitiş tekerlemeleri: "evvel zaman içinde,
           kalbur saman içinde", "develer tellal iken, pireler berber iken", "onlar ermiş muradına",
           "gökten üç elma düşmüş".
  talimat  Sesli duyuru (TDK: anons) açılışları: "sayın yolcularımız", "değerli müşterilerimiz",
           "dikkatinize sunarız", "anons"; yönerge açılışları "kullanım kılavuzu", "adım adım".
"""
from __future__ import annotations

import re
import unicodedata

_MASAL = re.compile(r"bir\s+varmış\s+bir\s+yokmuş|evvel\s+zaman\s+içinde|kalbur\s+saman\s+içinde|develer\s+tellal\s+iken|"
                    r"pireler\s+berber\s+iken|ermiş\s+muradına|gökten\s+üç\s+elma\s+düşmüş")
_DUYURU = re.compile(r"\b(?:sayın|değerli)\s+(?:yolcularımız|müşterilerimiz|ziyaretçilerimiz|velilerimiz|"
                     r"izleyicilerimiz|dinleyicilerimiz|seyircilerimiz|hemşehrilerimiz|vatandaşlarımız)\b|"
                     r"dikkatinize\s+sunarız|\banons\b")


def _kucuk(s: str) -> str:
    return unicodedata.normalize("NFC", s.replace("I", "ı").replace("İ", "i")).lower()


def kural_uslubu(metin: str) -> tuple[str, str] | None:
    k = _kucuk(metin)
    if (m := _MASAL.search(k)):
        return "masal", f"masal tekerlemesi ({m.group(0)})"
    if (m := _DUYURU.search(k[:200])):
        return "talimat", f"duyuru kalıbı ({m.group(0)})"
    return None
