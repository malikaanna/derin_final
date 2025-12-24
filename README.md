# Fruit Freshness: GAN vs VAE Karşılaştırma Projesi

## 📋 Proje Açıklaması
Bu proje, Fruit Freshness Classification veri seti üzerinde DCGAN ve VAE modellerini karşılaştırmaktadır.

## 🚀 Kurulum

### 1. Bağımlılıkları Yükle
```bash
pip install -r requirements.txt
```

### 2. Kaggle API Yapılandırması
```bash
# Kaggle'dan API token indirin: https://www.kaggle.com/settings
# ~/.kaggle/kaggle.json dosyasına yerleştirin
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 3. Veri Setini İndir
```bash
python utils/download_data.py
```

## 🏃 Eğitim

### VAE Eğitimi
```bash
python train_vae.py --epochs 50 --batch_size 32 --lr 0.0002
```

### DCGAN Eğitimi
```bash
python train_dcgan.py --epochs 50 --batch_size 32 --lr 0.0002
```

## 📊 Değerlendirme
```bash
python evaluate.py
python compare_models.py
```

## 📁 Proje Yapısı
```
├── data/                   # Veri seti
├── models/                 # Model tanımları
│   ├── vae.py
│   └── dcgan.py
├── utils/                  # Yardımcı fonksiyonlar
├── outputs/                # Üretilen görüntüler
├── checkpoints/            # Model ağırlıkları
└── notebooks/              # Jupyter notebook'ları
```

## 📈 Sonuçlar
Eğitim tamamlandıktan sonra `outputs/` klasöründe üretilen görüntüleri inceleyebilirsiniz.
