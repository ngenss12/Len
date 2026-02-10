# AIS_Classification1 — LSTM (Paper-based) for Fishing Activity Classification

Repo ini dibuat khusus untuk dataset AIS Studio (Baidu) **“public dataset including three fishing activities”**
yang dipakai di paper Ocean Engineering 303 (2024) 117711.

## Konsep sesuai paper
- Setiap **trajectory** berupa deretan titik dengan **interval 2 menit** dan **4 fitur**: longitude, latitude, speed, course.  
  (lihat penjelasan fitur & interval 2 menit di paper) fileciteturn10file4L44-L47
- Label 3 kelas (public dataset):
  - **0 = seine (围网)**
  - **1 = trawler (拖网)**
  - **2 = gillnet (刺网)** fileciteturn10file4L18-L21
- Metrik evaluasi yang digunakan: **Accuracy, Recall, Precision, F1-score, ROC/AUC** fileciteturn10file9L1-L4

> Catatan: Paper yang kamu kirim adalah model MFGTN (Transformer multi-modal). Di sini kita implement **baseline Deep Learning LSTM**
yang lebih ringan dan mudah dijalankan, tapi metrik output mengikuti yang paper pakai.

## Struktur folder (SAMAIN dengan screenshot kamu)
```
AIS_Classification1/
  train/          <- taruh file train_dataset dari Baidu di sini
  test_dataset/   <- taruh file test_dataset dari Baidu di sini
  submit_example.csv
  train.py
  predict.py
  src/
```

## Instalasi
Di Windows (Git Bash / CMD / PowerShell), dari folder `AIS_Classification1` jalankan:

```bash
py -m pip install -r requirements.txt
```

Kalau error `pip: command not found`:
```bash
py -m ensurepip --upgrade
py -m pip install -r requirements.txt
```

## Training + hasil akurasi (%)
Jalankan:
```bash
py train.py --train_dir train --out_dir outputs/lstm --epochs 30 --batch_size 64 --max_len 512
```

Output yang akan muncul per epoch:
- `acc=...%` (akurasi persen)
- `prec(macro)=...%`
- `recall(macro)=...%`
- `f1(macro)=...%`
- (kalau bisa dihitung) `auc(macro-ovr)=...`

Hasil terbaik tersimpan di:
```
outputs/lstm/best/
  model.pt
  scaler.pkl
  best_metrics.json
  classification_report.txt
  confusion_matrix.json
```

## Prediksi test_dataset → submission CSV
Jalankan:
```bash
py predict.py --test_dir test_dataset --model_dir outputs/lstm/best --submit_template submit_example.csv --out_csv infer/submit.csv
```

Output:
- `infer/submit.csv` (format sama seperti `submit_example.csv`)
- `infer/submit_proba.npy` (probability 3 kelas)

## Tips penting
- Kalau train data kamu **sudah** interval 2 menit, biarkan `--resample_mins 0`.
  Kalau belum, kamu bisa nyalakan:
  ```bash
  py train.py --resample_mins 2 ...
  py predict.py --resample_mins 2 ...
  ```

Selamat jalanin. Kalau ada error “kolom tidak ketemu”, kirim 5 baris head dari file CSV-nya (tanpa sensor data sensitif).
