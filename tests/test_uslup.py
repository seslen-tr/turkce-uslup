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


def test_yonerge_dizisi():                         # G&K §21.4.4.2: yönergeler kibar emir kipiyle
    assert uslup("Tencereye iki su bardağı su ekleyin. Kısık ateşte pişirin.").uslup == "talimat"
    assert uslup("Ardından karışımı fırına verin.").uslup == "talimat"           # sıra sözü + emir
    assert uslup("Formu doldurun. Ardından imzalayıp teslim edin.").uslup == "talimat"


def test_kisi_eki_emir_degil():                    # -sInIz, -DIn: emir ekine benzer ama kişi ekidir
    from turkce_uslup.kurallar import kural_uslubu
    assert kural_uslubu("Bazılarınız beni kandırdığınızı sanıyorsunuz. Ama yanılıyorsunuz.") is None
    assert kural_uslubu("Dün akşam ne yaptınız? Sinemaya gittiniz. Eve döndünüz.") is None
    assert kural_uslubu("Bekleyin! Ben size göstereceğim.") is None             # konuşmadaki buyruk
