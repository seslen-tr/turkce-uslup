# -*- coding: utf-8 -*-
"""
VERİ — üslup (metin türü) eğitim verisini indirir ve tekrarlanabilir biçimde böler.  python egit/veri.py

Üslup ↔ kaynak (hepsi açık lisanslı, sürümü sabit):
  haber    UD Turkish-BOUN "news" (ulusal gazeteler)                       CC BY-SA 4.0
  bilgi    UD Turkish-BOUN "bio" (biyografi) + Türkçe Vikipedi              CC BY-SA 4.0
  talimat  UD Turkish-BOUN "ins" (kullanım kılavuzu, yönerge)               CC BY-SA 4.0
  deneme   UD Turkish-BOUN "ess" (deneme) + "pop" (popüler kültür yazısı)   CC BY-SA 4.0
  anlati   Common Voice Türkçe cümle derlemi (çoğu roman ve öyküden)        CC0
  sohbet   XED Türkçe film altyazıları (konuşma dili)                       CC BY 4.0
BOUN cümleleri Türkçe Ulusal Derlem'den alınmıştır; tür, cümle kimliğinin önekindedir (news_12, bio_7…).
Her üslup en çok 3000 cümleyle sınırlanır (sınıf dengesi). Bölme: cümlenin MD5 özetine göre
%80 eğitim, %10 doğrulama, %10 sınav. Çıktı veri/{egitim,dogrulama,sinav}.tsv: cümle<TAB>üslup<TAB>kaynak
"""
from __future__ import annotations

import csv
import hashlib
import io
import random
import re
import sys
import urllib.request
from pathlib import Path

KOK = Path(__file__).resolve().parents[1]
VERI = KOK / "veri"
USLUPLAR = ("haber", "bilgi", "talimat", "deneme", "anlati", "sohbet")
SINIR = 3000

BOUN_SURUM = "f828b02872123a8dfb336584eab0767c4e55c5f9"
BOUN_URL = "https://raw.githubusercontent.com/UniversalDependencies/UD_Turkish-BOUN/{}/tr_boun-ud-{}.conllu"
BOUN_TUR = {"news": "haber", "bio": "bilgi", "ins": "talimat", "ess": "deneme", "pop": "deneme"}
XED_SURUM = "76666d251f147cf518071bef4f8415dab6b96115"
XED_URL = f"https://raw.githubusercontent.com/Helsinki-NLP/XED/{XED_SURUM}/Projections/tr-projections2.tsv"
CV_SURUM = "b12cde653757295103c5948c1cd601b166e1b905"
CV_URL = f"https://raw.githubusercontent.com/common-voice/common-voice/{CV_SURUM}/server/data/tr/sentence-collector.txt"


def temizle(c: str) -> str:
    return " ".join(re.sub(r"^[\s\-–—]+", "", c).split())


def bolum(cumle: str) -> str:
    h = int(hashlib.md5(cumle.encode("utf-8")).hexdigest(), 16) % 10
    return "sinav" if h == 0 else "dogrulama" if h == 1 else "egitim"


def _indir(url: str) -> str:
    return urllib.request.urlopen(url, timeout=120).read().decode("utf-8")


def boun() -> list[tuple[str, str, str]]:
    out = []
    for parca in ("train", "dev", "test"):
        kimlik = ""
        for s in _indir(BOUN_URL.format(BOUN_SURUM, parca)).splitlines():
            if s.startswith("# sent_id"):
                kimlik = s.split("=", 1)[1].strip()
            elif s.startswith("# text") and kimlik:
                tur = BOUN_TUR.get(kimlik.split("_")[0])
                if tur:
                    out.append((temizle(s.split("=", 1)[1]), tur, f"boun-{kimlik.split('_')[0]}"))
    return out


def vikipedi(adet: int) -> list[tuple[str, str, str]]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    yol = hf_hub_download("wikimedia/wikipedia", "20231101.tr/train-00001-of-00002.parquet", repo_type="dataset")
    metin = pq.read_table(yol, columns=["text"]).column("text").to_pylist()
    random.Random(7).shuffle(metin)
    out = []
    for m in metin:
        for c in re.split(r"(?<=[.!?])\s+", m.split("\n")[0].strip()):
            if 5 <= len(c.split()) <= 30 and c[:1].isupper() and c.endswith(".") and not re.search(r"[|=<>{}\[\]]", c):
                out.append((temizle(c), "bilgi", "vikipedi"))
                break
        if len(out) >= adet:
            break
    return out


def xed(adet: int) -> list[tuple[str, str, str]]:
    satir = [p[0] for p in list(csv.reader(io.StringIO(_indir(XED_URL)), delimiter="\t"))[1:] if p and p[0].strip()]
    satir = [temizle(c) for c in satir if len(c.split()) >= 3]
    random.Random(7).shuffle(satir)
    return [(c, "sohbet", "xed") for c in satir[:adet]]


def common_voice(adet: int) -> list[tuple[str, str, str]]:
    satir = [temizle(c) for c in _indir(CV_URL).splitlines() if len(c.split()) >= 4]
    random.Random(7).shuffle(satir)
    return [(c, "anlati", "commonvoice") for c in satir[:adet]]


def main() -> int:
    VERI.mkdir(exist_ok=True)
    hepsi = boun()
    bilgi_boun = sum(1 for _, u, _ in hepsi if u == "bilgi")
    hepsi += vikipedi(max(0, SINIR - bilgi_boun)) + xed(SINIR) + common_voice(SINIR)
    gorulen: set[str] = set()
    sinifli: dict[str, list] = {u: [] for u in USLUPLAR}
    for c, u, k in hepsi:
        if c and c not in gorulen:
            gorulen.add(c)
            sinifli[u].append((c, u, k))
    bolumler: dict[str, list] = {"egitim": [], "dogrulama": [], "sinav": []}
    for u, satir in sinifli.items():
        random.Random(11).shuffle(satir)
        for c, u_, k in satir[:SINIR]:
            bolumler[bolum(c)].append((c, u_, k))
    for ad, satir in bolumler.items():
        with open(VERI / f"{ad}.tsv", "w", encoding="utf-8", newline="\n") as f:
            for c, u, k in satir:
                f.write(f"{c}\t{u}\t{k}\n")
        sayim = {u: sum(1 for _, x, _ in satir if x == u) for u in USLUPLAR}
        print(f"{ad}.tsv {len(satir)}  {sayim}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
