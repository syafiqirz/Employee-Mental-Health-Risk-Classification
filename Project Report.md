# Laporan Proyek Machine Learning  - Muhammad Syafiq Irzaky
# Prediksi Risiko Kesehatan Mental Pekerja Menggunakan Machine Learning

# Domain Proyek

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

## 🧠 Data Understanding

### 📁 Deskripsi Dataset

Dataset yang digunakan dalam proyek ini berasal dari [Kaggle Mental Health Dataset](https://www.kaggle.com/datasets/mahdimashayekhi/mental-health). Dataset ini merupakan hasil simulasi sintetis dari survei global mengenai kesehatan mental di tempat kerja yang melibatkan 10.000 responden. Data ini mencerminkan pola nyata berdasarkan survei publik terkait kesehatan mental, namun telah dianonimkan sepenuhnya untuk menjamin privasi.

### 🎯 Tujuan dan Kegunaan Dataset

Dataset ini dirancang untuk mendukung berbagai eksperimen dalam analisis kesehatan mental di tempat kerja, di antaranya:
- Memprediksi kemungkinan seseorang mencari bantuan profesional.
- Menganalisis faktor-faktor yang memengaruhi tingkat stres dan risiko mental.
- Melatih model klasifikasi atau clustering untuk segmentasi risiko kesehatan mental.
- Membuat dashboard realistis untuk analitik HR atau sistem kesehatan.

Dataset ini ideal untuk pelatihan model machine learning, eksplorasi fairness, dan pembuatan sistem prediksi berbasis risiko kesehatan mental.

### ⚙️ Data yang Digunakan

Dari total 10.000 data, proyek ini difokuskan pada **pekerja saja** (`employment_status = Employed`) sehingga data yang dianalisis adalah sebanyak **5.868 baris**. Setelah difilter, kolom `employment_status` dihapus karena seluruh nilainya homogen.

### 📊 Struktur Dataset

Berikut adalah informasi awal dari dataset berdasarkan fungsi `df.info()` dan `df.describe()`:

- Terdapat 13 kolom (fitur), terdiri dari tipe data kategorikal (object) dan numerikal (int64, float64).
- Tidak ada nilai yang hilang di dalam dataset (semua baris lengkap).
- Rentang nilai numerik cukup masuk akal dan realistis, contohnya:
  - `stress_level`: 1 – 10
  - `depression_score`: 0 – 30
  - `productivity_score`: 42.8 – 100

### 🧾 Deskripsi Fitur

#### 🏷️ Fitur Kategorikal

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

#### 🔢 Fitur Numerikal

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

### 🔍 Ringkasan Eksplorasi (Exploratory)
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

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

