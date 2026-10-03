from turkce_uslup import OKUMA, okuma_ayari, uslup


def test_masal_tekerlemesi():                      # TDK: "bir varmış bir yokmuş" masal tekerlemesi
    r = uslup("Bir varmış bir yokmuş, evvel zaman içinde bir padişah varmış.")
    assert r.uslup == "masal" and r.ayar == OKUMA["masal"]


def test_duyuru():
    assert uslup("Sayın yolcularımız, uçağımız birazdan iniş yapacaktır.").uslup == "talimat"


def test_stuyo_ayari_degisir():
    a = okuma_ayari("haber", hiz=1.2)
    assert a.hiz == 1.2 and a.tonlama == OKUMA["haber"].tonlama


def test_model_metin():
    metin = ("Hey, nereye gidiyorsun öyle? Dur bir dakika, ben de geliyorum. Tamam, acele etme. "
             "Hadi ama, geç kalacağız. Peki peki, geliyorum işte.")
    assert uslup(metin).uslup == "sohbet"
