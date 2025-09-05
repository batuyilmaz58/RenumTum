# RenumTum 🏥

**RenumTum**, böbrek taşı ve böbrek tümörü tespiti için geliştirilmiş gelişmiş bir masaüstü uygulamasıdır. YOLOv8m modelini kullanan bu uygulama, tıbbi görüntülerde anlık tespit yaparak sağlık profesyonellerine destek sağlar.

## ✨ Özellikler

- 🎯 **Yüksek Doğruluk**: RTX 5090 ile eğitilmiş YOLOv8m modeli
- 📊 **Zengin Veri Seti**: 8K görüntü ile eğitilmiş güçlü model
- 🔄 **Çift Model Sistemi**: Böbrek taşı ve tümör için özelleştirilmiş 2 ayrı model
- ⚡ **Anlık Tespit**: Gerçek zamanlı görüntü analizi
- 🖥️ **Modern UI**: CustomTkinter ile tasarlanmış kullanıcı dostu arayüz
- 💾 **Veri Saklama**: JSON formatında yerel veri depolama
- 🔧 **Profesyonel Araçlar**: Ultralytics ve Supervision kütüphaneleri

## 🛠️ Teknolojiler

- **Programlama Dili**: Python
- **ML Framework**: YOLOv8m (Ultralytics)
- **UI Framework**: CustomTkinter
- **Computer Vision**: Supervision
- **Veri Formatı**: JSON
- **Eğitim Donanımı**: RTX 5090

## 📋 Sistem Gereksinimleri

- Python 3.8+
- NVIDIA GPU (CUDA desteği önerilir)
- Minimum 8GB RAM
- Windows/Linux/macOS

## 🚀 Kurulum

1. **Repository'yi klonlayın:**
   ```bash
   git clone https://github.com/kullaniciadi/renumtum.git
   cd renumtum
   ```

2. **Virtual environment oluşturun:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # veya
   venv\Scripts\activate     # Windows
   ```

3. **Gerekli paketleri yükleyin:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Uygulamayı çalıştırın:**
   ```bash
   python main.py
   ```

## 📖 Kullanım

1. **Uygulama Başlatma**: `main.py` dosyasını çalıştırarak uygulamayı başlatın
2. **Görüntü Yükleme**: Analiz edilecek tıbbi görüntüyü uygulamaya yükleyin
3. **Model Seçimi**: Böbrek taşı veya tümör tespiti için uygun modeli seçin
4. **Analiz**: Tespit işlemini başlatın ve sonuçları anlık olarak görüntüleyin
5. **Sonuçları Kaydetme**: Tespit sonuçları otomatik olarak JSON formatında kaydedilir

## 📁 Proje Yapısı

```
RenumTum/
├── main.py                 # Ana uygulama dosyası
├── models/                 # Eğitilmiş YOLOv8 modelleri
│   ├── kidney_stone.pt
│   └── kidney_tumor.pt
├── ui/                     # CustomTkinter UI dosyaları
├── utils/                  # Yardımcı fonksiyonlar
├── data/                   # JSON veri dosyaları
├── requirements.txt        # Python bağımlılıkları
└── README.md              # Bu dosya
```

## 🔧 Gerekli Paketler

```txt
ultralytics
customtkinter
supervision
opencv-python
```

## 📊 Model Bilgileri

- **Model Tipi**: YOLOv8m (Medium)
- **Eğitim Veri Seti**: 8K görüntü
- **Eğitim Donanımı**: RTX 5090
- **Tespit Sınıfları**: Böbrek taşı, Böbrek tümörü
- **Çıktı Formatı**: Bounding box + Güven skoru

## 🖼️ Ekran Görüntüleri

![UI_predict_1](/uı_predict/UI_Predict_1.png)
![UI_predict_2](/uı_predict/UI_Predict_2.png)
![UI_predict_3](/uı_predict/UI_predict_3.png)

## 🤝 Katkıda Bulunma

1. Bu repository'yi fork edin
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add amazing feature'`)
4. Branch'inizi push edin (`git push origin feature/amazing-feature`)
5. Pull Request oluşturun

## ⚠️ Uyarılar

- Bu uygulama eğitim ve araştırma amaçlıdır
- Tıbbi kararlar için mutlaka uzman hekim görüşü alınmalıdır
- Sonuçlar yalnızca yardımcı bilgi amaçlı kullanılmalıdır

## 📄 Lisans

Bu proje [MIT Lisansı](LICENSE) altında lisanslanmıştır.

## 📞 İletişim

- **Geliştirici**: Batuhan Yılmaz
- **E-posta**: batuhanyilmaz0011@gmail.com
- **GitHub**: [github.com/batuyilmaz58]

---

**Not**: RenumTum, tıbbi görüntü analizi alanında AI teknolojilerinin potansiyelini göstermek amacıyla geliştirilmiştir. Profesyonel tıbbi teşhis için mutlaka uzman hekim desteği alınmalıdır.
