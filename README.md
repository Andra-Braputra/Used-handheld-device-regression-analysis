# Used Handheld Device — Regression Analysis

Notebook ini berisi analisis regresi lengkap untuk memprediksi harga jual HP bekas (`normalized_used_price`) menggunakan dataset dari Kaggle. Proyek ini mencakup preprocessing, eksplorasi data, berbagai pendekatan encoding, pelatihan model, evaluasi K-Fold, stress test, hingga tuning hyperparameter.

---

## Dataset

**Sumber:** [Kaggle — Used Handheld Device Data](https://www.kaggle.com/datasets/ahsan81/used-handheld-device-data)  
**File:** `used_device_data.csv`  
**Target:** `normalized_used_price`

---

## Struktur Notebook

### 1. Preprocessing

- Import library dan dataset via `kagglehub`
- Deteksi dan penanganan _missing values_ menggunakan **Median Imputation** (dan kode alternatif **KNN Imputer** tersedia di komentar)
- _Exploratory Data Analysis_ (EDA): distribusi target, correlation heatmap
- Encoding fitur kategorikal:
  - **One Hot Encoding** → fitur `os`, `4g`, `5g`
  - **Target Encoding** (sesudah split) → fitur `device_brand` _(aktif)_
  - Alternatif tersedia di komentar: Target Encoding sebelum split, One Hot Encoding brand, Label Encoding
- Konversi kolom boolean ke integer
- Train-Test Split **80:20** dengan `random_state=42`
- **StandardScaler** — fit hanya pada data training

---

### 2. Baseline Test

| Model                      | Catatan                            |
| -------------------------- | ---------------------------------- |
| Baseline Linear Regression | Model dasar OLS tanpa transformasi |
| Baseline Lasso (alpha=0.1) | Model Lasso tanpa optimasi target  |

Fungsi evaluasi `evaluate_model()` menghitung **R², MAE, RMSE, MAPE**.

---

### 3. Code Improvement

Model ditingkatkan dengan **log-transformation pada target** (`y_train_log = np.log1p(y_train)`) dan prediksi di-inverse dengan `np.expm1()`.

| Model                        | Konfigurasi                                             |
| ---------------------------- | ------------------------------------------------------- |
| Ridge Regression (Improved)  | `alpha=1.0`, target log-transformed                     |
| Lasso Regression (Improved)  | `alpha=0.001`, `max_iter=10000`, target log-transformed |
| Polynomial Linear Regression | `degree=2`, tanpa log-transform                         |
| Polynomial Ridge             | `degree=2`, `alpha=1.0`, target log-transformed         |

**K-Fold Cross Validation (5 fold)** dilakukan untuk semua model di atas menggunakan `Pipeline` agar tidak terjadi data leakage pada scaling.

---

### 4. Stress Test

Empat pengujian ketahanan model Polynomial Linear Regression:

| #   | Nama                       | Deskripsi                                             |
| --- | -------------------------- | ----------------------------------------------------- |
| 1   | Extreme Distribution Shift | Input ±3 std dari mean training                       |
| 2   | Noise Injection            | Tambah noise Gaussian 5% pada test set                |
| 3   | Outlier Sensitivity        | Evaluasi setelah menghapus outlier IQR dari training  |
| 4   | Top 5 Error Analysis       | Identifikasi 5 prediksi dengan error absolut terbesar |

---

### 5. Experiment — GridSearch Alpha

Pencarian hyperparameter `alpha` terbaik menggunakan **GridSearchCV (5 fold)** dengan metrik RMSE.

| Model | Rentang Alpha                                  | Alpha Terbaik |
| ----- | ---------------------------------------------- | ------------- |
| Ridge | `[0.001, 0.01, 0.1, 1, 10, 50, 100]`           | **0.001**     |
| Lasso | `[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 1]` | **0.0001**    |

Alpha terbaik yang sangat kecil mengkonfirmasi bahwa dataset ini tidak membutuhkan regularisasi kuat.

---

## Hasil Terbaik

| Model                              | R²         | MAE        | RMSE       | MAPE       |
| ---------------------------------- | ---------- | ---------- | ---------- | ---------- |
| **Polynomial Linear Regression** ★ | **0.8521** | **0.1746** | **0.2191** | **0.0408** |
| Polynomial Ridge                   | 0.8469     | 0.1770     | 0.2229     | 0.0413     |
| Baseline Linear Regression         | 0.8364     | 0.1850     | 0.2304     | 0.0436     |
| Ridge / Lasso (Improved)           | ~0.8188    | ~0.1925    | ~0.2424    | ~0.0453    |
| Baseline Decision Tree             | 0.6989     | 0.2405     | 0.3126     | 0.0559     |

---

## Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
kagglehub
```

Install:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn kagglehub
```

---

## Cara Menjalankan

1. Pastikan akun Kaggle sudah terhubung (`kaggle.json` tersedia)
2. Jalankan notebook secara berurutan dari sel pertama
3. Pastikan tidak melewati sel preprocessing karena variabel digunakan di sel berikutnya

---

## Catatan

- Encoding aktif: **Target Encoding Sesudah Split** + **One Hot Encoding** untuk `os`, `4g`, `5g`
- Beberapa metode encoding alternatif tersedia dalam bentuk komentar dan bisa diaktifkan untuk eksperimen
- Log-transformation pada target (`np.log1p`) secara signifikan meningkatkan performa model linear
