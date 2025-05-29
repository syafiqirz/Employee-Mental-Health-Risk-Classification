# Laporan Proyek Machine Learning - Prediksi Risiko Kesehatan Mental Pekerja
### Oleh Muhammad Syafiq Irzaky

# 🌍 Domain Proyek

## 1. Latar Belakang

Kesehatan mental merupakan aspek penting dalam kehidupan manusia, termasuk dalam konteks dunia kerja. Gangguan kesehatan mental di lingkungan kerja dapat muncul akibat tekanan pekerjaan, beban kerja yang tinggi, serta gaya hidup yang tidak sehat. Dalam industri teknologi, permasalahan ini menjadi semakin relevan mengingat tingginya ekspektasi performa dan dinamika kerja yang cepat. Sayangnya, isu kesehatan mental pekerja seringkali kurang mendapat perhatian karena tidak tampak secara fisik dan sering kali tidak disadari oleh manajemen perusahaan (Sebayang, Chrisnanto, & Melina, 2023).

Studi oleh Alfarezy, Ermatita, dan Wadu (2022) menegaskan bahwa kesadaran terhadap pentingnya kesehatan mental di lingkungan kerja masih tergolong rendah. Oleh karena itu, organisasi seperti Open Source Mental Illness (OSMI) berinisiatif melakukan survei untuk mengukur kesadaran dan kondisi kesehatan mental pekerja, terutama di industri teknologi. Data dari survei ini menunjukkan bahwa gangguan kesehatan mental memiliki pengaruh signifikan terhadap produktivitas dan kesejahteraan pekerja, serta menyarankan perlunya intervensi berbasis data.

Lebih lanjut, Firmansyah dan Yulianto (2024) menunjukkan bahwa penerapan algoritma pembelajaran mesin dapat menjadi solusi strategis untuk memprediksi kondisi kesehatan mental karyawan secara preventif. Dengan memanfaatkan model seperti Gradient Boost, Decision Tree, dan Naïve Bayes, perusahaan dapat memperoleh prediksi yang cukup akurat untuk mengidentifikasi potensi risiko gangguan mental sejak dini.

## 2. Mengapa dan Bagaimana Masalah Ini Harus Diselesaikan

Gangguan kesehatan mental tidak hanya berdampak pada individu pekerja, tetapi juga berimplikasi pada penurunan produktivitas, peningkatan absensi, turnover tinggi, dan memburuknya lingkungan kerja secara keseluruhan. Berdasarkan temuan dari Sebayang et al. (2023), kesehatan mental yang terganggu akan memengaruhi kinerja individu dalam kontribusinya terhadap perusahaan. Oleh karena itu, penting bagi perusahaan untuk memiliki kepekaan dan sistem deteksi dini terhadap kondisi mental karyawan.

Solusi teknologi berbasis _machine learning_ memungkinkan perusahaan untuk secara proaktif dan prediktif mengidentifikasi potensi risiko kesehatan mental pekerja. Model klasifikasi seperti Random Forest (Sebayang et al., 2023), Naïve Bayes (Alfarezy et al., 2022), hingga Gradient Boost (Firmansyah & Yulianto, 2024) terbukti dapat mengklasifikasikan kondisi mental dengan akurasi yang cukup baik—antara 72% hingga 84%. Model-model ini dapat diintegrasikan dalam sistem informasi sumber daya manusia (HRIS) untuk mendukung pengambilan keputusan berbasis data dan merancang program intervensi yang sesuai dengan kebutuhan karyawan.

Implementasi sistem prediksi ini tidak hanya bersifat reaktif, tetapi juga menjadi pendekatan preventif yang efektif dalam menciptakan lingkungan kerja yang sehat secara psikologis. Dengan demikian, perusahaan tidak hanya meningkatkan kesejahteraan karyawan, tetapi juga memastikan keberlanjutan dan efektivitas organisasi secara keseluruhan.

## 3. Referensi

- Alfarezy, R., Ermatita, E., & Wadu, R. M. B. (2022). Implementasi Algoritma Naïve Bayes Untuk Analisis Klasifikasi Survei Kesehatan Mental (Studi Kasus: Open Sourcing Mental Illness). _Prosiding Seminar Nasional Mahasiswa Bidang Ilmu Komputer dan Aplikasinya_, 3(2).  
- Firmansyah, F., & Yulianto, A. (2024). Pemodelan Pembelajaran Mesin untuk Prediksi Kesehatan Mental di Tempat Kerja. _Jurnal Minfo Polgan_, 13(1), 397–407.  
- Sebayang, E. R. B., Chrisnanto, Y. H., & Melina, M. (2023). Klasifikasi Data Kesehatan Mental di Industri Teknologi Menggunakan Algoritma Random Forest. _IJESPG (International Journal of Engineering, Economic, Social Politic and Government)_, 1(3), 237–253.  

# 📈 Business Understanding

## 1. Problem Statements (Pernyataan Masalah)
Kesehatan mental pekerja merupakan aspek krusial yang berpengaruh langsung terhadap performa dan produktivitas perusahaan. Namun, banyak organisasi belum memiliki sistem yang mampu mengidentifikasi risiko gangguan kesehatan mental secara dini, terutama dalam sektor teknologi yang sangat dinamis dan menuntut. Akibatnya, gangguan kesehatan mental sering terlambat ditangani, menyebabkan peningkatan absensi, _burnout_, penurunan produktivitas, hingga _turnover_ yang tinggi.

Pernyataan masalah utama yang diangkat dalam proyek ini adalah:  
- **Bagaimana** membangun model _machine learning_ yang mampu memprediksi risiko gangguan kesehatan mental pekerja dengan akurasi yang tinggi dan dapat diandalkan sebagai alat bantu pengambilan keputusan di lingkungan kerja?  
- **Bagaimana** model ini dapat digunakan untuk meningkatkan kesadaran dan intervensi dini terhadap kondisi mental karyawan dalam rangka mendukung keberlangsungan performa organisasi?  

## 2. Goals (Tujuan)
Tujuan dari proyek ini adalah:  
- Mengembangkan model prediksi risiko kesehatan mental pekerja berdasarkan data yang mencakup fitur-fitur seperti demografi, tingkat stres, riwayat kesehtaan mental, akses terhadap dukungan mental, dll.  
- Mengidentifikasi dan mengklasifikasikan pekerja yang berpotensi mengalami gangguan kesehatan mental.  
- Menyediakan alat bantu bagi tim Human Resources (HR) atau konseling untuk melakukan tindakan preventif dan strategis.  
- Meningkatkan kesadaran perusahaan terhadap pentingnya kondisi mental karyawan sebagai bagian dari keberlangsungan dan efektivitas bisnis.  

## 3. Solution Statement (Pernyataan Solusi)
Untuk mencapai tujuan di atas, berikut adalah dua solusi yang dirancang secara sistematis dan terukur:

### **Solusi 1: Implementasi dan Perbandingan Beberapa Algoritma Klasifikasi**  
Melatih dan membandingkan beberapa model _supervised learning_ berbasis klasifikasi menggunakan dataset dari sumber terbuka, seperti Kaggle:  

**Algoritma yang digunakan:**  
- K-Neirest Neighbor (KNN)
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- Naïve Bayes

**Metrik evaluasi:**  
- Accuracy  
- Precision  
- Recall  
- F1-score  

Model terbaik akan dipilih berdasarkan keseimbangan metrik di atas, terutama **F1-score** untuk menghindari _bias_ akibat _imbalance_ pada kelas (misalnya jumlah pekerja dengan risiko rendah vs risiko medium).  

### **Solusi 2: Peningkatan Model dengan Hyperparameter Tuning dan Feature Engineering**  
Setelah menentukan model _baseline_, dilakukan proses optimasi untuk meningkatkan performa model:  

**Langkah yang dilakukan:**  
- _Hyperparameter tuning_ menggunakan _Grid Search_ atau _Bayessian Search_ pada model terbaik (misalnya Random Forest atau KNN).  
- _Feature selection_ menggunakan metode seperti membandingkan nilai korelasi untuk menentukan fitur-fitur paling relevan.  

**Metrik evaluasi setelah tuning:**  
- Dibandingkan kembali menggunakan metrik yang sama untuk menilai peningkatan performa.  
- Penekanan pada peningkatan **recall** untuk memastikan sebanyak mungkin individu berisiko tinggi dapat terdeteksi.  

# 🧠 Data Understanding

## Deskripsi Dataset

Dataset yang digunakan dalam proyek ini berasal dari [Kaggle Mental Health Dataset](https://www.kaggle.com/datasets/mahdimashayekhi/mental-health). Dataset ini merupakan hasil simulasi sintetis dari survei global mengenai kesehatan mental di tempat kerja yang melibatkan 10.000 responden. Data ini mencerminkan pola nyata berdasarkan survei publik terkait kesehatan mental, namun telah dianonimkan sepenuhnya untuk menjamin privasi.

## Tujuan dan Kegunaan Dataset

Dataset ini dirancang untuk mendukung berbagai eksperimen dalam analisis kesehatan mental di tempat kerja, di antaranya:
- Memprediksi kemungkinan seseorang mencari bantuan profesional.
- Menganalisis faktor-faktor yang memengaruhi tingkat stres dan risiko mental.
- Melatih model klasifikasi atau clustering untuk segmentasi risiko kesehatan mental.
- Membuat dashboard realistis untuk analitik HR atau sistem kesehatan.

Dataset ini ideal untuk pelatihan model machine learning, eksplorasi fairness, dan pembuatan sistem prediksi berbasis risiko kesehatan mental.

## Data yang Digunakan

Dari total 10.000 data, proyek ini difokuskan pada **pekerja saja** (`employment_status = Employed`) sehingga data yang dianalisis adalah sebanyak **5.868 baris**. Setelah difilter, kolom `employment_status` dihapus karena seluruh nilainya homogen.

## Struktur Dataset

Berikut adalah informasi awal dari dataset berdasarkan fungsi `df.info()` dan `df.describe()`:

- Terdapat 13 kolom (fitur), terdiri dari tipe data kategorikal (object) dan numerikal (int64, float64).
- Tidak ada nilai yang hilang di dalam dataset (semua baris lengkap).
- Rentang nilai numerik cukup masuk akal dan realistis, contohnya:
  - `stress_level`: 1 – 10
  - `depression_score`: 0 – 30
  - `productivity_score`: 42.8 – 100

## Deskripsi Fitur

### Fitur Kategorikal

- **gender**  
  Jenis kelamin responden. Nilai: `Male`, `Female`, `Non-binary`, `Prefer not to say`.

- **work_environment**  
  Jenis lingkungan kerja. Nilai: `On-site`, `Remote`, `Hybrid`.

- **mental_health_history**  
  Apakah responden memiliki riwayat masalah kesehatan mental. Nilai: `Yes`, `No`.

- **seeks_treatment**  
  Apakah responden pernah mencari bantuan profesional. Nilai: `Yes`, `No`.

- **mental_health_risk**  
  Kategori risiko kesehatan mental yang dimiliki responden. Nilai: `Low`, `Medium`, `High`. Ini adalah **target variabel** yang akan diprediksi dalam proyek klasifikasi.

### Fitur Numerikal

- **age**  
  Usia responden dalam rentang 18–65 tahun.

- **stress_level**  
  Skor tingkat stres (1–10), semakin tinggi menunjukkan tingkat stres yang lebih tinggi.

- **sleep_hours**  
  Rata-rata jam tidur per hari (3–10 jam).

- **physical_activity_days**  
  Jumlah hari aktif secara fisik dalam satu minggu (0–7 hari).

- **depression_score**  
  Skor depresi (0–30), makin tinggi menunjukkan kondisi yang lebih parah.

- **anxiety_score**  
  Skor kecemasan (0–21), makin tinggi menunjukkan tingkat kecemasan yang lebih tinggi.

- **social_support_score**  
  Skor dukungan sosial yang diterima (0–100).

- **productivity_score**  
  Skor produktivitas kerja (0–100), makin tinggi makin baik.

## Ringkasan Eksplorasi (Exploratory)
- Dataset berisi 5868 baris data lengkap.
- Semua fitur memiliki tipe data sesuai konteks penggunaannya.
- Tidak terdapat missing values, sehingga tidak diperlukan penanganan imputasi.
- Usia rata-rata responden adalah 41.76 tahun.
- Rata-rata tingkat stres adalah 5.6 dengan standar deviasi sekitar 2.88.
- Rata-rata jam tidur per hari adalah 6.45 jam, dengan minimum 3 dan maksimum 10.
- Skor produktivitas rata-rata adalah 77.3 dengan maksimum 100.
- Tidak ada nilai hilang atau outlier signifikan.
- Fitur `depression_score`, `anxiety_score`, dan `productivity_score` menunjukkan korelasi kuat dengan target dan dipertahankan untuk model prediktif.
- Variabel kategorikal menunjukkan distribusi yang relatif seimbang, meskipun terdapat ketimpangan pada target `mental_health_risk`.

# ⚙️ Data Preparation

Tahapan data preparation dilakukan untuk memastikan bahwa data dalam kondisi optimal sebelum digunakan dalam proses modeling. Langkah-langkah dilakukan secara berurutan sebagai berikut:

## 1. Filter Gender yang Relevan
Hanya responden dengan gender "Male" dan "Female" yang disertakan dalam analisis. Data dengan gender seperti "Non-binary" atau "Prefer not to say" dihapus karena jumlahnya sangat kecil dan berpotensi menimbulkan noise dalam pemodelan. Setelah itu, indeks dataset di-reset agar lebih rapi.

## 2. Encoding Variabel Kategorikal
### Encoding Secara Manual

**Label Encoding** adalah teknik preprocessing data yang mengubah nilai kategorikal (seperti "Male"/"Female") menjadi nilai numerik (0/1). 

Tujuan utamanya:
1. Memungkinkan algoritma machine learning memproses data kategorikal
2. Mengkonversi teks menjadi format numerik yang bisa diolah komputer
3. Mempertahankan urutan jika variabel bersifat ordinal (misal: "Low"=0, "Medium"=1, "High"=2)

Pada project ini, encoding dilakukan secara manual menggunakan `map()` karena lebih sederhana, transparan, dan memberikan kontrol penuh terhadap urutan nilai numerik—terutama untuk fitur ordinal seperti `work_environment`. Selain itu, teknik ini efisien untuk variabel biner dan menghindari overhead tambahan dari library encoder.

Beberapa variabel kategorikal diubah menjadi representasi numerik:
- Variabel `gender` diubah menjadi format numerik agar dapat diproses oleh model machine learning:
  - Male → 0
  - Female → 1
- `work_environment`: dikodekan berdasarkan urutan logis dari fleksibel ke kaku:
  - Remote → 0
  - Hybrid → 1
  - On-site → 2
- `mental_health_history` dan `seeks_treatment`: diubah menjadi format biner (Yes → 1, No → 0)

## 3. Encoding Target: mental_health_risk
Kolom target `mental_health_risk` dikonversi ke format numerik agar bisa digunakan untuk klasifikasi:
- Low → 0
- Medium → 1
- High → 2
Kolom `mental_health_risk_encoded` yang sebelumnya dibuat manual kemudian dihapus karena sudah tidak diperlukan lagi.

## 4. Standarisasi Fitur Numerikal
Fitur numerik seperti `depression_score`, `anxiety_score`, dan `productivity_score` memiliki skala yang berbeda. Untuk menyamakan skala dan mencegah bias algoritma terhadap fitur tertentu, dilakukan standarisasi menggunakan `StandardScaler`, sehingga semua fitur memiliki:
- Rata-rata = 0
- Standar deviasi = 1

Standarisasi sangat penting terutama untuk algoritma berbasis jarak seperti KNN dan SVM.

## 5. Train-Test Split
Data dibagi menjadi dua bagian:
- **Training set (90%)**: untuk melatih model
- **Testing set (10%)**: untuk menguji performa model terhadap data yang belum pernah dilihat

Pemisahan dilakukan dengan parameter `random_state=123` agar hasilnya konsisten setiap kali dijalankan ulang.

Ukuran data setelah pembagian:
- Total data: 5.290 sampel
- Training set: 4.232 sampel
- Testing set: 1.058 sampel

Pembagian ini bertujuan untuk menghindari overfitting dan memungkinkan evaluasi model secara objektif.

# 🧮 Modeling

Tahap ini berfokus pada pembangunan model machine learning untuk mengklasifikasikan tingkat risiko kesehatan mental berdasarkan fitur-fitur yang telah diproses. Untuk memperoleh gambaran performa awal dan membandingkan efektivitas berbagai algoritma, digunakan **lima model klasifikasi populer** sebagai baseline dengan **hyperparameter default**. Tujuan utamanya adalah mengidentifikasi model terbaik untuk dioptimalkan lebih lanjut pada tahap selanjutnya.

## Model yang Digunakan

Berikut lima model klasifikasi yang digunakan:

1. **K-Nearest Neighbors (KNN)**  
   KNN mengklasifikasikan sampel baru berdasarkan mayoritas label dari `k` tetangga terdekat dalam ruang fitur. Model ini cocok digunakan karena data telah melalui proses standarisasi, yang sangat penting dalam algoritma berbasis jarak seperti KNN.

2. **Decision Tree (DT)**  
   DT membangun struktur pohon berdasarkan pembagian informasi (information gain) untuk memisahkan kelas target. Algoritma ini mampu menangkap interaksi antar fitur dan memberikan interpretasi yang jelas terhadap proses klasifikasi.

3. **Random Forest (RF)**  
   RF adalah algoritma ensemble yang membangun banyak pohon keputusan dan menggabungkan prediksinya untuk meningkatkan akurasi dan mengurangi overfitting. Cocok untuk dataset dengan fitur heterogen dan kompleks, serta tahan terhadap outlier.

4. **Support Vector Machine (SVM)**  
   SVM bekerja dengan mencari hyperplane terbaik yang memisahkan kelas dalam ruang fitur berdimensi tinggi. Penggunaan SVM didukung oleh standarisasi fitur, karena SVM sangat sensitif terhadap skala data.

5. **Naive Bayes (NB)**  
   NB adalah model probabilistik yang mengasumsikan independensi antar fitur. Meskipun sederhana, model ini sering memberikan hasil yang kompetitif, terutama jika data sudah bersih dan variabel relevan.

## Alasan Penggunaan Banyak Model

Penggunaan lima model ini bertujuan untuk:
- **Membandingkan performa awal (baseline)** dari berbagai pendekatan klasifikasi.
- **Mengidentifikasi model yang paling sesuai** dengan karakteristik data.
- Menjadi dasar dalam proses **seleksi dan tuning model terbaik** pada tahap selanjutnya.

Seluruh model dilatih menggunakan data training dan diuji dengan data testing yang telah dipisahkan sebelumnya. Evaluasi dilakukan dengan metrik akurasi, precision, recall, dan f1-score untuk memperoleh gambaran menyeluruh terhadap performa model.

# 💯 Evaluation

Tahap evaluasi bertujuan untuk menilai performa model dalam mengklasifikasikan tingkat risiko kesehatan mental berdasarkan fitur-fitur yang tersedia. Evaluasi dilakukan menggunakan metrik standar untuk kasus klasifikasi, yaitu **akurasi**, **precision**, **recall**, dan **F1-score**. Selain itu, **confusion matrix** digunakan untuk melihat distribusi prediksi model terhadap kelas-kelas yang ada.

## Metrik Evaluasi yang Digunakan

- **Akurasi** mengukur proporsi prediksi yang benar dari keseluruhan prediksi.
```
Accuracy = (True Positive + True Negative) / (True Positive + True Negative + False Positive + False Negative)
```

- **Precision** menghitung seberapa banyak prediksi positif yang benar.
```
Precision = True Positive / (True Positive + False Positive)
```

- **Recall** mengukur kemampuan model dalam menangkap seluruh data positif yang benar.
```
Recall = True Positive / (True Positive + False Negative)
```

- **F1-Score** adalah rata-rata harmonik dari precision dan recall, memberikan keseimbangan di antara keduanya.
```
F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
```

## Hasil Evaluasi pada Data Latih

| Model             | Akurasi | Precision (avg)  | Recall (avg)  | F1-score (avg) |
|-------------------|---------|------------------|---------------|----------------|
| KNN               | 0.96    | 0.96             | 0.96          | 0.96           |
| Decision Tree     | 1.00    | 1.00             | 1.00          | 1.00           |
| Random Forest     | 1.00    | 1.00             | 1.00          | 1.00           |
| SVM               | 0.99    | 0.99             | 0.99          | 0.99           |
| Naive Bayes       | 0.88    | 0.87             | 0.87          | 0.87           |

**Interpretasi:**
Model seperti **Decision Tree** dan **Random Forest** mencapai akurasi 100% pada data latih, mengindikasikan kemungkinan **overfitting**, yakni ketika model terlalu menghafal data tanpa kemampuan generalisasi yang baik. Model **SVM** dan **KNN** juga menunjukkan performa tinggi (masing-masing 99% dan 96%), sementara **Naive Bayes** sedikit lebih rendah (88%).

## Evaluasi pada Data Uji

| Model              | Akurasi | Precision (avg) | Recall (avg) | F1-score (avg) |
|-------------------|---------|------------------|---------------|----------------|
| KNN               | 0.94    | 0.94             | 0.92          | 0.93           |
| Decision Tree     | 1.00    | 1.00             | 1.00          | 1.00           |
| Random Forest     | 0.98    | 0.98             | 0.98          | 0.98           |
| SVM               | 0.98    | 0.98             | 0.98          | 0.98           |
| Naive Bayes       | 0.87    | 0.86             | 0.85          | 0.85           |

**Interpretasi:**

- **Decision Tree** menunjukkan akurasi sempurna bahkan di data uji, namun ini jarang terjadi dan mengindikasikan kemungkinan **overfitting**, meskipun hasilnya sangat baik.

- **Random Forest** dan **SVM** menampilkan hasil sangat tinggi dan konsisten (98%), menjadikannya kandidat kuat karena mampu menangkap pola kompleks sekaligus menjaga generalisasi yang baik.

- **KNN** menurun sedikit pada recall, terutama di kelas minoritas, namun tetap menunjukkan generalisasi yang solid dengan akurasi 94%.

- **Naive Bayes** mencatat akurasi terendah, kemungkinan karena asumsi independensi antar fitur tidak sepenuhnya berlaku pada data ini.

# 📋 Kesimpulan

Berdasarkan evaluasi:
- **Decision Tree** memiliki akurasi sempurna, baik saat dites menggunakan data latih maupun data uji. Perlu ditinjau lebih lanjut karena berpotensi overfitting.
- **Random Forest** dan **SVM** menjadi dua alternatif model terbaik karena menghasilkan metrik yang tinggi, seimbang, dan konsisten.

## Rekomendasi:
- Model **Decision Tree** cocok digunakan jika data memiliki dimensionalitas (jumlah fitur) dan varians antar fitur yang rendah. Model ini mudah dipahami dan diinterpretasikan, tetapi performanya dapat menurun secara signifikan ketika dihadapkan dengan data yang memiliki variansi tinggi atau mengandung banyak noise.
- Model **Random Forest** adalah pilihan yang lebih seimbang dan umumnya lebih andal. Model ini menghasilkan akurasi yang tinggi dan konsisten, serta cukup robust terhadap data berdimensionalitas tinggi maupun varians fitur yang besar. Selain itu, Random Forest lebih toleran terhadap outlier dan noise kecil, sehingga cocok digunakan untuk data dunia nyata yang sering kali tidak bersih atau memiliki ketidakteraturan.
- Sama seperti **Random Forest**, **SVM** juga menawarkan performa yang baik dan dapat menangani data dengan margin yang kompleks. Namun, model ini relatif lebih kompleks untuk dilatih, terutama pada dataset besar dan berdimensionalitas tinggi. Selain itu, SVM cenderung kurang toleran terhadap outlier dan perbedaan distribusi antara data latih dan data aktual di lapangan, yang dapat menyebabkan penurunan performa jika data tidak distandarkan atau dibersihkan dengan baik.
