# -*- coding: utf-8 -*-
"""
KURALLAR — modelin göremediği ya da veride olmayan, kalıplaşmış açılışlar.

  masal    TDK: "bir varmış bir yokmuş" — "bir masala başlarken, 'eskiden' anlamında söylenen bir
           tekerleme". Aynı geleneğin öteki başlangıç ve bitiş tekerlemeleri: "evvel zaman içinde,
           kalbur saman içinde", "develer tellal iken, pireler berber iken", "onlar ermiş muradına",
           "gökten üç elma düşmüş".
  talimat  Sesli duyuru (TDK: anons) açılışları: "sayın yolcularımız", "değerli müşterilerimiz",
           "dikkatinize sunarız", "anons"; yönerge açılışları "kullanım kılavuzu", "adım adım".
  talimat  Yönerge dizisi: tarif, kılavuz ve yönergeler 2. çoğul / kibar emir kipiyle yazılır
           (-(y)In, -(y)InIz; G&K §21.4.4.2). En az iki cümle ve cümlelerin yarısı bu kiple bitiyorsa ya da
           bir cümle sıra sözüyle (önce, ardından, daha sonra, son olarak…) başlayıp bu kiple bitiyorsa.
           Yalnız noktayla biten cümleler: konuşmadaki buyruklar ("Bekleyin!") çoğunlukla ünlemle biter.
           Model burada zayıf: eğitimdeki talimat örnekleri tek kaynaktan (UD BOUN "ins") geliyor.
"""
from __future__ import annotations

import re
import unicodedata

_MASAL = re.compile(r"bir\s+varmış\s+bir\s+yokmuş|evvel\s+zaman\s+içinde|kalbur\s+saman\s+içinde|develer\s+tellal\s+iken|"
                    r"pireler\s+berber\s+iken|ermiş\s+muradına|gökten\s+üç\s+elma\s+düşmüş")
_DUYURU = re.compile(r"\b(?:sayın|değerli)\s+(?:yolcularımız|müşterilerimiz|ziyaretçilerimiz|velilerimiz|"
                     r"izleyicilerimiz|dinleyicilerimiz|seyircilerimiz|hemşehrilerimiz|vatandaşlarımız)\b|"
                     r"dikkatinize\s+sunarız|\banons\b")


_EMIR = re.compile(r"(?:[aeıioöuü]y)?[ıiuü]n(?:[ıiuü]z)?$")
# Aynı sesle biten, emir olmayan sık sözcükler (ad, sıfat, edat, ilgi/tamlayan eki almış sık adlar)
_EMIR_DEGIL = {"gün", "dün", "bütün", "için", "uzun", "kadın", "yakın", "akın", "ekin", "keskin", "yorgun",
               "olgun", "durgun", "dalgın", "kalın", "sakin", "zengin", "bugün", "düğün", "salgın", "yangın",
               "sonun", "benim", "senin", "onun", "bizim", "sizin", "kimin", "neyin", "hangisinin", "derin",
               "serin", "engin", "yazın", "kışın", "sabahın", "akşamın", "gecenin"}
_SIRA = re.compile(r"^(?:önce|ilk\s+olarak|ilk\s+önce|öncelikle|ardından|daha\s+sonra|sonra|son\s+olarak|en\s+son)\b")


# Emir ekine benzeyen kişi ekleri: -sIn(Iz) (yanılıyorsunuz, olacaksın, etsin), geçmiş zaman -DIn(Iz)
# (demiştin, kurdunuz). Geçmiş zamanı "edin / gidin" gibi kısa emirlerden uzunluk ayırır.
_KISI = re.compile(r"s[ıiuü]n(?:[ıiuü]z)?$")
_GECMIS = re.compile(r"[dt][ıiuü]n(?:[ıiuü]z)?$")
_KISA_EMIR = {"edin", "ediniz", "gidin", "gidiniz"}


def _emirle_biter(cumle: str) -> bool:
    w = re.findall(r"[a-zçğıöşüâîû]+", cumle)
    if not w or cumle.rstrip()[-1:] not in (".", ""):
        return False
    s = w[-1]
    if s in _KISA_EMIR:
        return True
    if len(s) <= 3 or s in _EMIR_DEGIL or not _EMIR.search(s) or _KISI.search(s):
        return False
    return not (_GECMIS.search(s) and len(s) > 5)


def _kucuk(s: str) -> str:
    return unicodedata.normalize("NFC", s.replace("I", "ı").replace("İ", "i")).lower()


def kural_uslubu(metin: str) -> tuple[str, str] | None:
    k = _kucuk(metin)
    if (m := _MASAL.search(k)):
        return "masal", f"masal tekerlemesi ({m.group(0)})"
    if (m := _DUYURU.search(k[:200])):
        return "talimat", f"duyuru kalıbı ({m.group(0)})"
    cumleler = [c.strip() for c in re.split(r"(?<=[.!?…])\s+", k) if c.strip()]
    emir = [c for c in cumleler if _emirle_biter(c)]
    if len(emir) >= 2 and len(emir) * 2 >= len(cumleler):
        return "talimat", f"yönerge dizisi ({len(emir)}/{len(cumleler)} cümle emir kipiyle bitiyor)"
    if any(_SIRA.match(c) for c in emir):
        return "talimat", "sıra sözü + emir kipi"
    return None
