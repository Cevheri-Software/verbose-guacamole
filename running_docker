Güzel, şu an iki tane Docker konteynerin çalışıyor:

| Container ID   | İmaj Adı                            | Durum          | İsim                  |
| -------------- | ----------------------------------- | -------------- | --------------------- |
| `86e5fc8743ae` | `nvcr.io/nvidia/l4t-ml:r35.2.1-py3` | **Up 19 mins** | `wizardly_black`      |
| `767706ad4905` | `nvcr.io/nvidia/l4t-base:r32.4.3`   | **Up 2 hours** | `youthful_mcclintock` |

---

## ✅ Ne Yapabilirsin?

### 1. 📥 `jett.py` dosyasını konteynıra kopyalamak

Eğer `jett.py` dosyan **host (Jetson)** makinedeyse ve örneğin şu yoldaysa:

```bash
/home/cevheri/jett.py
```

o zaman şu komutla `wizardly_black` adlı aktif konteynıra kopyalayabilirsin:

```bash
sudo docker cp /home/cevheri/jett.py wizardly_black:/workspace/jett.py
```

> Konteynerin içinde `/workspace` klasöründe olacaktır.

---

### 2. 🔧 Konteynıra girip dosyayı çalıştırmak

```bash
sudo docker exec -it wizardly_black bash
```

Artık içindesin. Sonra:

```bash
cd /workspace
ls
python3 jett.py
```

---

## ✅ Alternatif Yöntem: Klasörü bağlamak (daha otomatik)

Bir dahaki sefere konteyneri başlatırken dosyanın bulunduğu klasörü mount et:

```bash
sudo docker run -it --rm \
  --runtime nvidia \
  --gpus all \
  --network host \
  --device /dev/video0 \
  -v /home/cevheri:/workspace \
  -w /workspace \
  nvcr.io/nvidia/l4t-ml:r35.2.1-py3
```

Bu sayede host'taki `/home/cevheri/jett.py` konteynerin içinden `/workspace/jett.py` olarak görünür.

---

## 🔄 Özet

| Amaç                                             | Komut                                                 |
| ------------------------------------------------ | ----------------------------------------------------- |
| Aktif konteynıra dosya kopyala                   | `sudo docker cp ~/jett.py wizardly_black:/workspace/` |
| Konteynıra gir                                   | `sudo docker exec -it wizardly_black bash`            |
| Python dosyasını çalıştır                        | `python3 /workspace/jett.py`                          |
| Klasörü otomatik bağla (bir dahaki çalıştırmada) | `-v /home/cevheri:/workspace`                         |

---

İstersen adım adım birlikte de yapabiliriz. Yüklediğin dosyanın yolunu ve çalışmak istediğin komutu gönder, doğrudan hazırlayayım.
