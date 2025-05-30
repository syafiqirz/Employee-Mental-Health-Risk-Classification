# -*- coding: utf-8 -*-
"""Mental Health Risk Classification.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1kz_Jo0ulzYcpR5imcGzmPbo8ETaiZY_q

# 🧠 Prediksi Risiko Kesehatan Mental Pekerja Menggunakan Machine Learning

Notebook ini bertujuan untuk membangun model machine learning yang dapat memprediksi risiko gangguan kesehatan mental pada pekerja, khususnya di industri teknologi. Permasalahan ini diangkat karena rendahnya kesadaran dan sistem deteksi dini terhadap kondisi mental karyawan, yang berdampak pada:
- Produktivitas
- Tingkat absensi
- Turnover yang tinggi

# Import Library
"""

import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report

"""Pada bagian ini, dilakukan proses import library yang diperlukan untuk analisis data dan pengembangan model machine learning. Library seperti pandas, numpy, dan matplotlib.pyplot digunakan untuk manipulasi data dan visualisasi. seaborn mendukung visualisasi yang lebih informatif dan estetik. Peringatan yang tidak perlu disembunyikan menggunakan warnings.filterwarnings('ignore') agar output lebih bersih. Untuk pemodelan, digunakan berbagai algoritma klasifikasi dari scikit-learn, seperti K-Nearest Neighbors, Decision Tree, Random Forest, Support Vector Machine, dan Naïve Bayes. Selain itu, disertakan juga fungsi evaluasi model seperti accuracy_score, precision_score, recall_score, f1_score, dan confusion_matrix untuk mengukur performa model secara menyeluruh. Data juga akan diproses menggunakan StandardScaler dan dibagi menggunakan train_test_split agar model dapat dilatih dan diuji secara adil.

# Load Data

Pada sel ini, dataset terkait kesehatan mental pekerja diimpor dari repositori GitHub dalam format CSV menggunakan pandas.read_csv(). Dataset ini berisi informasi terkait atribut demografis, kondisi kerja, kebiasaan gaya hidup, serta skor terkait kesehatan mental dan produktivitas. Setelah data berhasil dimuat ke dalam DataFrame df, ditampilkan secara langsung untuk meninjau struktur dan isi awal dataset.
"""

url = 'https://raw.githubusercontent.com/syafiqirz/Employee-Mental-Health-Risk-Classification/refs/heads/main/mental_health_dataset%202.csv'
df = pd.read_csv(url)
df

"""Sel ini digunakan untuk mengetahui distribusi data berdasarkan kolom employment_status (status pekerjaan). Dengan menggunakan value_counts(), ditampilkan jumlah data untuk masing-masing kategori seperti Employed, Student, Self-employed, dan Unemployed. Informasi ini berguna untuk memahami proporsi data serta menentukan subset data yang akan dianalisis."""

df['employment_status'].value_counts()

"""Karena fokus proyek adalah pada prediksi risiko kesehatan mental pekerja, maka hanya data dengan status pekerjaan Employed yang dipertahankan. Data lainnya (misalnya mahasiswa atau pengangguran) dihapus. Setelah proses penyaringan, indeks DataFrame di-reset agar terurut kembali dari nol."""

df = df[df['employment_status'] == 'Employed']
df.reset_index(drop=True, inplace=True)
df

"""Sel ini digunakan untuk memverifikasi bahwa hanya data dengan employment_status bernilai Employed yang tersisa setelah proses penyaringan. Output menunjukkan bahwa semua data sekarang berasal dari kategori pekerja aktif."""

df['employment_status'].value_counts()

"""Karena seluruh data kini berasal dari kelompok Employed, kolom employment_status sudah tidak lagi memberikan informasi tambahan yang berguna dan dihapus dari DataFrame. Penghapusan dilakukan menggunakan fungsi drop() dengan axis=1 yang berarti kolom, bukan baris."""

df = df.drop('employment_status', axis=1)
df

"""# 📊 Exploratory Data Analysis (EDA)

Pada tahap EDA ini, dilakukan eksplorasi awal terhadap data untuk memahami struktur, distribusi, dan karakteristik masing-masing variabel. EDA membantu dalam mengidentifikasi potensi masalah seperti data kosong, pencilan (outliers), atau distribusi yang tidak seimbang yang dapat mempengaruhi performa model. Proses ini merupakan langkah penting sebelum dilakukan pemodelan machine learning, karena kualitas input data akan sangat menentukan kualitas prediksi yang dihasilkan.

## Deskripsi Variabel

Sel ini digunakan untuk melihat informasi umum mengenai dataset, seperti nama kolom, tipe data, dan jumlah nilai tidak kosong pada setiap kolom. Hasil menunjukkan bahwa seluruh kolom memiliki 5868 baris tanpa nilai yang hilang (missing values) dan terdiri dari kombinasi tipe data numerik (int64, float64) dan kategorikal (object).
"""

df.info()

"""Fungsi describe() memberikan ringkasan statistik dari kolom numerik, seperti nilai minimum, maksimum, rata-rata, dan kuartil. Informasi ini sangat penting untuk memahami distribusi data dan skala masing-masing fitur. Misalnya, stress_level memiliki rentang 1–10, sementara depression_score berkisar antara 0–30. Hal ini juga memberi gambaran awal apakah ada kemungkinan data yang ekstrem"""

df.describe()

"""# Deskripsi Variabel Dataset Mental Health

## Fitur Kategorikal

**gender**  
Jenis kelamin responden (Male, Female, Non-binary, Prefer not to say)

**work_environment**  
Lingkungan kerja (On-site, Remote, Hybrid)

**mental_health_history**  
Riwayat masalah mental sebelumnya (Yes/No)

**seeks_treatment**  
Pernah mencari bantuan profesional (Yes/No)

**mental_health_risk**  
Tingkat risiko kesehatan mental (Low, Medium, High)

## Fitur Numerikal

**age**  
Usia responden (18-65 tahun)

**stress_level**  
Tingkat stres (skala 1-10)

**sleep_hours**  
Rata-rata jam tidur harian (3-10 jam)

**physical_activity_days**  
Hari aktif fisik per minggu (0-7 hari)

**depression_score**  
Skor depresi (0-30, makin tinggi makin parah)

**anxiety_score**  
Skor kecemasan (0-21)

**social_support_score**  
Tingkat dukungan sosial (0-100)

**productivity_score**  
Tingkat produktivitas (0-100, makin tinggi makin baik)

## Menangani Missing Value

Dari hasil df.info() sebelumnya dan sel di bawah ini, diketahui bahwa tidak ada nilai yang hilang pada dataset ini. Dengan demikian, tidak perlu dilakukan penanganan terhadap missing value, dan data dapat langsung digunakan untuk eksplorasi lanjutan dan pelatihan model.
"""

df.isnull().sum()

"""## Deteksi Outliers

Sel ini menampilkan boxplot untuk seluruh kolom numerik, yang digunakan untuk mendeteksi outlier (nilai ekstrem). Boxplot memudahkan visualisasi distribusi dan outlier. Dari hasil visualisasi, tidak ditemukan outlier yang mencolok pada variabel-variabel numerik seperti sleep_hours, stress_level, maupun productivity_score. Ini menunjukkan bahwa data memiliki distribusi yang relatif bersih dan tidak perlu dilakukan transformasi atau penghapusan nilai outlier.
"""

categorical_columns = df.select_dtypes(exclude=np.number).columns.tolist()
numerical_columns = df.select_dtypes(include=np.number).columns.tolist()

plt.figure(figsize=(15, 10))
plt.suptitle('Boxplot Variabel Numerik', y=1.02, fontsize=16)

for i, col in enumerate(numerical_columns, 1):
    plt.subplot(3, 3, i)  # 3 baris, 3 kolom, posisi ke-i
    sns.boxplot(data=df, y=col, color='skyblue', width=0.5)
    plt.title(col)
    plt.ylabel('')

plt.tight_layout()
plt.show()

"""## Analisis Univariat

Visualisasi ini menunjukkan distribusi frekuensi dari masing-masing variabel kategorikal dalam dataset. Tujuannya adalah untuk memahami bagaimana data tersebar pada tiap kategori dan apakah ada ketidakseimbangan yang signifikan yang perlu diperhatikan saat proses pemodelan.

### Variabel Kategorikal
"""

plt.figure(figsize=(15, 15))
plt.suptitle('Distribusi Fitur Kategorikal', y=1.02, fontsize=20)
for i, col in enumerate(categorical_columns, 1):
    plt.subplot(3, 3, i)
    ax = sns.countplot(data=df, x=col, palette='viridis',
                      order=df[col].value_counts().index)

    total = len(df)
    for p in ax.patches:
        percentage = f'{100 * p.get_height()/total:.1f}%'
        ax.annotate(percentage,
                   (p.get_x() + p.get_width()/2., p.get_height()),
                   ha='center', va='center',
                   xytext=(0, 5),
                   textcoords='offset points')

    plt.title(f'Distribusi {col}', pad=10)
    plt.xticks(rotation=45)
    plt.xlabel('')

plt.tight_layout()
plt.show()

"""### Distribusi Variabel Kategorikal

#### **gender**:
Terdapat distribusi yang hampir seimbang antara responden laki-laki (45.2%) dan perempuan (44.9%). Sisanya adalah responden yang memilih identitas non-biner (5.2%) atau tidak ingin menyebutkan gender mereka (4.7%).

#### **work_environment**:
Mayoritas responden bekerja secara on-site (50.2%), diikuti oleh yang bekerja secara remote (29.9%), dan hybrid (19.9%). Ini menunjukkan mayoritas data berasal dari lingkungan kerja fisik.

#### **mental_health_history**:
Sebanyak 70.0% responden tidak memiliki riwayat gangguan kesehatan mental, sementara 30.0% lainnya memiliki riwayat tersebut. Proporsi ini penting diperhatikan saat mengevaluasi risiko kesehatan mental.

#### **seeks_treatment**:
Sebagian besar responden (59.9%) tidak mencari pengobatan, sementara 40.1% lainnya melaporkan pernah mencari bantuan atau pengobatan terkait kesehatan mental.

#### **mental_health_risk**:
Distribusi tingkat risiko kesehatan mental menunjukkan bahwa sebagian besar responden dikategorikan sebagai risiko sedang (Medium) sebesar 58.8%, diikuti oleh risiko tinggi (High) 23.9%, dan risiko rendah (Low) 17.3%. Distribusi ini menunjukkan adanya ketidakseimbangan kelas yang mungkin perlu ditangani pada tahap pemodelan klasifikasi.

### Variabel Numerikal
"""

plt.figure(figsize=(18, 12))

for i, col in enumerate(numerical_columns, 1):
    plt.subplot(3, 3, i)
    sns.histplot(data=df, x=col, kde=True, bins=30, color='skyblue')

    plt.title(f'Distribusi {col}', fontsize=12, pad=10)
    plt.xlabel(col, fontsize=10)
    plt.ylabel('Frekuensi', fontsize=10)
    plt.grid(axis='y', alpha=0.3)

    mean_val = df[col].mean()
    median_val = df[col].median()
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=1, label=f'Mean: {mean_val:.1f}')
    plt.axvline(median_val, color='green', linestyle='-', linewidth=1, label=f'Median: {median_val:.1f}')
    plt.legend()

plt.tight_layout()
plt.suptitle('Distribusi Variabel Numerik', y=1.02, fontsize=16)
plt.show()

"""### Distribusi Fitur Numerikal

#### **age**  
Distribusi usia cukup merata antara 18 hingga 65 tahun, dengan rata-rata 41.8 dan median 42.0. Ini menunjukkan sebaran usia yang luas tanpa dominasi kelompok usia tertentu.

#### **stress_level**  
Variabel ini memiliki nilai antara 1 hingga 10, dengan distribusi mendekati seragam. Rata-rata (5.6) dan median (6.0) menunjukkan bahwa sebagian besar individu mengalami tingkat stres sedang.

#### **sleep_hours**  
Sebaran cenderung normal dengan puncak di sekitar 6–7 jam tidur per malam. Rata-rata dan median identik di angka 6.4, menunjukkan distribusi simetris.

#### **physical_activity_days**  
Menunjukkan jumlah hari aktif secara fisik per minggu (0–7 hari). Median (4 hari) lebih tinggi dari rata-rata (3.5 hari), menandakan sedikit skew ke kiri (lebih banyak individu dengan sedikit hari aktif).

#### **depression_score**  
Skor ini berkisar 0–30 dengan distribusi hampir seragam. Mean dan median berada di tengah (15.1 dan 15.0), menunjukkan tidak adanya kemencengan signifikan.

#### **anxiety_score**  
Distribusi cenderung merata dengan puncak kecil di skor menengah. Rata-rata (10.5) sedikit lebih rendah dari median (11.0), menunjukkan sedikit skew ke kiri.

#### **social_support_score**  
Skor ini berkisar dari 0 hingga 100. Distribusinya hampir seragam, dengan rata-rata (50.1) dan median (50.0), menandakan distribusi yang sangat simetris.

#### **productivity_score**  
Distribusi sedikit miring ke kiri, dengan lebih banyak nilai tinggi (80+). Rata-rata dan median berada di sekitar 77.3, mencerminkan mayoritas responden memiliki tingkat produktivitas tinggi.

## Analisis Multivariat

## Analisis Multivariat: Keterkaitan Fitur Kategorikal dengan mental_health_risk

Visualisasi berikut memperlihatkan bagaimana risiko kesehatan mental (`mental_health_risk`) terdistribusi dalam setiap kategori dari variabel kategorikal. Tujuannya adalah untuk melihat pola dan hubungan potensial antara karakteristik individu dengan tingkat risiko kesehatan mental.
"""

# Tentukan jumlah kolom per baris
cols = 2
rows = math.ceil(len(categorical_columns) / cols)

plt.figure(figsize=(15, 5 * rows))
for i, col in enumerate(categorical_columns, 1):
    plt.subplot(rows, cols, i)
    sns.countplot(data=df, x=col, hue='mental_health_risk', palette='Set3')
    plt.title(f"Distribusi 'mental_health_risk' relatif terhadap - {col}")
    plt.xticks(rotation=45)  # Jika label sumbu X panjang
plt.tight_layout()
plt.show()

"""### Penjelasan Tiap Grafik:

1. **`gender` vs `mental_health_risk`**  
   Menunjukkan perbedaan tingkat risiko berdasarkan jenis kelamin. Terlihat bahwa mayoritas responden perempuan memiliki risiko sedang. Jumlah risiko tinggi juga tampak lebih banyak pada perempuan dibandingkan laki-laki secara absolut.

2. **`work_environment` vs `mental_health_risk`**  
   Responden yang bekerja di lingkungan on-site cenderung memiliki risiko sedang. Sementara itu, pekerja remote memiliki distribusi risiko yang lebih seimbang, dengan proporsi risiko tinggi yang mencolok.

3. **`mental_health_history` vs `mental_health_risk`**  
   Responden yang memiliki riwayat kesehatan mental sebelumnya cenderung memiliki risiko tinggi yang lebih besar dibandingkan yang tidak memiliki riwayat.

4. **`seeks_treatment` vs `mental_health_risk`**  
   Responden yang pernah mencari pengobatan menunjukkan tingkat risiko tinggi yang lebih tinggi daripada yang belum pernah mencari bantuan.

5. **`mental_health_risk` (self-grouped)**  
   Menampilkan distribusi total dari seluruh risiko. Mayoritas responden dikategorikan dalam risiko sedang, dengan kelompok risiko tinggi dan rendah yang relatif lebih kecil.

## Analisis Multivariat: Pairplot Antar Variabel Numerik

Visualisasi berikut menggunakan **pairplot** untuk memetakan hubungan antar seluruh fitur numerik dalam dataset. Pairplot menampilkan scatter plot untuk setiap pasangan fitur dan **distribusi (KDE)** di diagonal.

### Tujuan:
- Melihat pola hubungan antar fitur numerik secara visual.
- Mengidentifikasi potensi **korelasi linear** atau non-linear.
- Mengamati bentuk distribusi setiap fitur numerik.
"""

map = {
    'Low': 0,
    'Medium': 1,
    'High': 2
}
df['mental_health_risk_encoded'] = df['mental_health_risk'].map(map)

sns.pairplot(df, diag_kind = 'kde')

"""### Insight dari Visualisasi:
- **`depression_score`** dan **`productivity_score`** memperlihatkan pola **korelasi negatif yang kuat**, sejalan dengan hasil korelasi sebelumnya.
- Korelasi antara **`depression_score`** dan **`anxiety_score`** terlihat dari sebaran yang cenderung membentuk pola linear positif.
- Fitur seperti `age`, `sleep_hours`, `stress_level`, dan `physical_activity_days` tampaknya memiliki distribusi cukup datar atau tidak menunjukkan pola hubungan jelas dengan fitur lain secara visual.
- Variabel target `mental_health_risk_encoded` tidak terlalu menunjukkan pemisahan visual yang kuat, namun nilai-nilainya terkonsentrasi pada kategori diskrit, sesuai ekspektasi.

## Analisis Multivariat: Korelasi antar Variabel Numerik

Visualisasi ini menampilkan **matriks korelasi** antara seluruh variabel numerik, termasuk `mental_health_risk` yang telah di-encode menjadi nilai numerik (`Low=0`, `Medium=1`, `High=2`).

### Tujuan:
Untuk memahami sejauh mana hubungan linear antar fitur, serta mengidentifikasi fitur yang mungkin berkorelasi kuat dengan `mental_health_risk`.
"""

numerical_columns = df.select_dtypes(include=np.number).columns.tolist()

correlation=df[numerical_columns].corr()
plt.figure(figsize =(12, 12))
plt.title("Correlation Matrix untuk Fitur Numerik ", size=20)
sns.heatmap(correlation, cbar=True, square=True, fmt='.2f', annot=True, annot_kws={'size':16}, cmap='spring')

"""### Interpretasi Korelasi:
- Nilai korelasi berkisar antara **-1 hingga 1**:
  - **1** = Korelasi positif sempurna
  - **-1** = Korelasi negatif sempurna
  - **0** = Tidak ada korelasi linear

### Insight Penting dari Visualisasi:
- **`depression_score`** menunjukkan korelasi positif yang cukup kuat (**0.71**) terhadap `mental_health_risk_encoded`, artinya semakin tinggi skor depresi, semakin tinggi risiko kesehatan mental.
- **`anxiety_score`** juga memiliki korelasi positif (**0.55**) terhadap risiko, memperkuat hubungan antara kecemasan dan risiko mental.
- **`productivity_score`** berkorelasi negatif tinggi (**-0.67**) terhadap risiko, menunjukkan bahwa individu dengan risiko tinggi cenderung memiliki produktivitas yang lebih rendah.
- Korelasi antara fitur lainnya tergolong **lemah hingga sangat lemah**, seperti `age`, `sleep_hours`, dan `physical_activity_days`.

### Kesimpulan:
Fitur seperti `depression_score`, `anxiety_score`, dan `productivity_score` merupakan kandidat kuat untuk fitur prediktor karena memiliki hubungan yang cukup signifikan dengan risiko kesehatan mental. Oleh karena itu, fitur numerik lain akan dihapus dan ketiga variabel tersebut akan tetap lanjut digunakan untuk proyek.
"""

df.drop(['age', 'stress_level', 'sleep_hours', 'physical_activity_days', 'social_support_score'], inplace=True, axis=1)
df.head()

"""# Data Preparation

### Filter Gender yang Relevan
Kita membatasi data hanya pada responden dengan gender "Male" dan "Female" untuk menyederhanakan analisis.

Responden dengan jawaban seperti "Non-binary" atau "Prefer not to say" dihapus karena representasi yang sangat kecil.

Index direset ulang agar rapi dan terstruktur kembali.
"""

df = df[(df['gender'] == 'Male') | (df['gender'] == 'Female')]
df.reset_index(drop=True, inplace=True)
df

"""## Encoding Variabel Kategorikal

### Encoding Gender
- Kolom gender dikonversi ke format numerik agar dapat digunakan dalam algoritma machine learning
- Encoding: Male → 0, Female → 1
"""

df['gender'].value_counts()

map = {
    'Male': 0,
    'Female': 1
}
df['gender'] = df['gender'].map(map)

"""### Encoding Variabel Kategorikal: Work Environment
- Kolom work_environment diubah menjadi angka berdasarkan urutan logis (Remote < Hybrid < On-site)
- Hal ini memungkinkan model mengenali pola numerik dari tipe lingkungan kerja terhadap risiko kesehatan mental
"""

df['work_environment'].value_counts()

map = {
    'Remote': 0,
    'Hybrid': 1,
    'On-site': 2

}
df['work_environment'] = df['work_environment'].map(map)

"""### Encoding Riwayat Kesehatan Mental dan Seeks Treatment
- Variabel mental_health_history dan seeks_treatment di-encode ke format biner (0/1)
- Ini diperlukan agar algoritma ML dapat memproses data kategorikal tersebut secara numerik
"""

df['mental_health_history'].value_counts()

df['seeks_treatment'].value_counts()

map = {
    'No': 0,
    'Yes': 1
}

df['mental_health_history'] = df['mental_health_history'].map(map)
df['seeks_treatment'] = df['seeks_treatment'].map(map)

"""### Encoding Target: mental_health_risk
- Kolom target mental_health_risk dikonversi ke angka agar bisa digunakan dalam klasifikasi
- Skala numerik: Low → 0, Medium → 1, High → 2
- Kolom mental_health_risk_encoded sebelumnya dibuat manual dan sekarang dihapus karena sudah tidak diperlukan
"""

df['mental_health_risk'].value_counts()

map = {
    'Low': 0,
    'Medium': 1,
    'High': 2
}
df['mental_health_risk'] = df['mental_health_risk'].map(map)

df = df.drop('mental_health_risk_encoded', axis=1)

"""### Cek info dataset
Menampilkan ringkasan struktur dataset setelah preprocessing.

Semua variabel sekarang bertipe numerik (int atau float) dan siap digunakan untuk tahap modeling selanjutnya.
"""

df.info()

"""## Standarisasi Variabel Numerikal

Variabel numerik `depression_score`, `anxiety_score`, dan `productivity_score` memiliki skala yang berbeda.

Kita menggunakan `StandardScaler` untuk melakukan standarisasi, yaitu mengubah distribusi data menjadi memiliki:
- Rata-rata = 0
- Standar deviasi = 1

Ini penting agar algoritma machine learning (terutama yang berbasis jarak seperti KNN dan SVM) dapat bekerja optimal tanpa bias terhadap skala variabel.
"""

numerical_columns = ['depression_score', 'anxiety_score', 'productivity_score']

scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
df.describe().round(2)

"""## Train Test Split"""

X = df.drop(["mental_health_risk"],axis =1)
y = df["mental_health_risk"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

print(f'Total # of sample in whole dataset: {len(X)}')
print(f'Total # of sample in train dataset: {len(X_train)}')
print(f'Total # of sample in test dataset: {len(X_test)}')

"""# Membangun Model

Setelah proses preprocessing yang mencakup encoding dan standardisasi fitur, serta pembagian data menjadi data latih dan uji (80:20), dilakukan pelatihan model menggunakan lima algoritma klasifikasi yang umum digunakan. Tujuan dari langkah ini adalah untuk membandingkan performa masing-masing model dalam mengklasifikasikan risiko kesehatan mental pekerja berdasarkan fitur-fitur yang tersedia.

## 1. K-Nearest Neighbors (KNN)
Algoritma ini mengklasifikasikan data baru berdasarkan kemiripan dengan data tetangga terdekat. Sangat bergantung pada nilai parameter k dan skala fitur, sehingga standardisasi yang telah dilakukan sangat penting.
"""

KNN = KNeighborsClassifier()
KNN.fit(X_train, y_train)

"""## 2. Decision Tree (DT)
Model ini membangun struktur pohon keputusan berdasarkan pembagian informasi untuk memisahkan kelas target. Mudah diinterpretasikan dan dapat menangkap interaksi antar fitur.
"""

DT = DecisionTreeClassifier()
DT.fit(X_train, y_train)

"""## 3. Random Forest (RF)
Merupakan ensemble dari banyak decision tree. Model ini cenderung lebih stabil dan memiliki generalisasi yang baik dibandingkan satu pohon keputusan.
"""

RF = RandomForestClassifier()
RF.fit(X_train, y_train)

"""## 4. Support Vector Machine (SVM)
Algoritma ini mencari hyperplane terbaik yang memisahkan kelas target. Cocok untuk data berdimensi tinggi dan telah distandarisasi seperti pada kasus ini.
"""

SVM = SVC()
SVM.fit(X_train, y_train)

"""## 5. Naive Bayes (NB)
Model probabilistik berbasis Teorema Bayes yang mengasumsikan independensi antar fitur. Meskipun sederhana, seringkali efektif terutama untuk data yang bersih dan terstandarisasi.
"""

NB = GaussianNB()
NB.fit(X_train, y_train)

"""# Evaluasi Model"""

models = {
    "KNN": KNN,
    "Decision Tree": DT,
    "Random Forest": RF,
    "SVM": SVM,
    "Gaussian Naive Bayes": NB
}

"""## Evaluasi Model dengan Dataset Latih

Setiap model diuji menggunakan data latih (training set) untuk melihat seberapa baik mereka mengenali pola yang telah dipelajari. Evaluasi dilakukan menggunakan classification report (berisi precision, recall, f1-score, dan akurasi) serta confusion matrix untuk melihat detail prediksi setiap kelas.
"""

for name, model in models.items():
    y_pred = model.predict(X_train) # Dataset latih
    cm = confusion_matrix(y_train, y_pred)
    report = classification_report(y_train, y_pred)

    print(f"==== {name} Classifier ====")
    print("\nClassification Report:")
    print(report)

    # Visualisasi Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    print("-" * 100 + "\n")

"""### Interpretasi Hasil

1. **Model K-Nearest Neighbors (KNN)** menunjukkan kinerja yang sangat baik dengan akurasi sekitar 96%. Model ini mampu mengklasifikasikan sebagian besar data latih dengan benar. Namun, karena KNN sangat bergantung pada kedekatan data dan skala, ada kemungkinan performa menurun jika data uji memiliki distribusi berbeda atau outlier.
2. **Model Decision Tree** mencapai akurasi sempurna (100%) pada data latih. Ini menunjukkan bahwa kedua model ini sangat cocok dengan data, bahkan bisa jadi terlalu cocok. Hasil ini mengindikasikan potensi overfitting, yaitu ketika model menghafal data latih dan mungkin tidak mampu melakukan generalisasi dengan baik terhadap data baru (data uji).
3. **Model Random Forest**, yang merupakan kumpulan (ensemble) dari banyak pohon keputusan, juga menunjukkan akurasi 100%. Ini menunjukkan bahwa Random Forest mampu mengenali pola dalam data latih dengan sangat baik. Namun, seperti Decision Tree, performa yang terlalu tinggi ini bisa menandakan overfitting. Meskipun Random Forest lebih tahan terhadap overfitting dibanding satu pohon, tetap perlu diuji pada data uji untuk memastikan model tidak hanya menghafal data latih.
4. **Model Support Vector Machine (SVM)** juga menunjukkan performa sangat tinggi dengan akurasi 99%. Tidak seperti pohon keputusan, SVM bekerja dengan mencari hyperplane terbaik dalam ruang fitur dan cenderung lebih tahan terhadap overfitting. Hasil ini menunjukkan bahwa SVM dapat menjadi kandidat kuat untuk model akhir karena mampu menangkap pola kompleks dengan generalisasi yang baik.
5. **Model Gaussian Naive Bayes** memiliki performa terendah di antara kelima model, dengan akurasi sekitar 88%. Walaupun cukup baik, model ini diasumsikan fitur-fitur input saling independen, yang dalam kenyataannya tidak selalu terjadi pada data kesehatan mental. Hal ini mungkin menjadi penyebab performanya tidak setinggi model lain.

### Kesimpulan
Secara umum, semua model menunjukkan kinerja yang sangat tinggi pada data latih, dengan pengecualian Gaussian Naive Bayes. Model Decision Tree dan Random Forest tampil sempurna, tetapi kemungkinan besar mengalami overfitting. SVM dan KNN menunjukkan performa tinggi dengan potensi generalisasi yang lebih baik. Untuk memastikan kemampuan generalisasi model, langkah selanjutnya adalah melakukan evaluasi pada data uji, yang akan memberikan gambaran nyata seberapa baik model dapat bekerja dalam situasi yang belum pernah dilihat sebelumnya.

## Evaluasi Model dengan Dataset Uji

Setelah mengevaluasi model pada data latih, langkah selanjutnya adalah menguji performa kelima model pada data uji. Evaluasi pada data uji penting untuk mengetahui seberapa baik model dapat menggeneralisasi terhadap data baru yang belum pernah dilihat sebelumnya. Pada tahap ini, digunakan metrik evaluasi berupa classification report (precision, recall, f1-score, dan akurasi), serta visualisasi confusion matrix untuk memberikan gambaran lebih jelas terhadap prediksi model.
"""

for name, model in models.items():
    y_pred = model.predict(X_test) # Data uji
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"==== {name} Classifier ====")
    print("\nClassification Report:")
    print(report)

    # Visualisasi Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    print("-" * 100 + "\n")

"""### Interpretasi Hasil

1. **Model K-Nearest Neighbors (KNN)** menunjukkan performa yang sangat baik pada data uji dengan akurasi sebesar 94%. Meskipun ada sedikit penurunan dibandingkan data latih (96%), hal ini masih wajar dan mengindikasikan bahwa KNN memiliki kemampuan generalisasi yang cukup baik. Penurunan recall pada kelas 0 (dari 94% menjadi 87%) menunjukkan bahwa model kadang keliru mengenali kelas tersebut.
2. **Model Decision Tree** kembali menunjukkan akurasi 100%, bahkan pada data uji. Ini adalah hasil yang sangat jarang dan menimbulkan indikasi kuat adanya overfitting — model tampaknya sangat cocok terhadap data yang diberikan, bahkan pada data uji. Meski hasilnya mengesankan, ini perlu diuji lebih lanjut pada data yang benar-benar baru atau cross-validation untuk memastikan performa yang stabil.
3. **Model Random Forest**menghasilkan akurasi 98% pada data uji. Meskipun tidak sempurna seperti Decision Tree, hasil ini jauh lebih realistis dan mengindikasikan bahwa model mampu menangkap pola kompleks tanpa berlebihan. Performanya sangat konsisten di semua kelas, menunjukkan stabilitas dan kehandalan tinggi dalam klasifikasi risiko kesehatan mental.
4. **Model Support Vector Machine (SVM)** juga tampil sangat baik dengan akurasi 98%, mirip dengan Random Forest. SVM menunjukkan distribusi metrik yang seimbang di seluruh kelas. Dengan f1-score tinggi dan sedikit perbedaan antara precision dan recall, SVM terlihat sebagai salah satu model paling stabil dan tidak overfit, cocok untuk deployment jangka panjang.
5. **Model Gaussian Naive Bayes** mencatat akurasi 87%, sedikit lebih rendah dari model lainnya. Meski begitu, ini tetap performa yang layak. Namun, dibanding model lain, terlihat bahwa prediksi GNB lebih lemah dalam mengklasifikasikan kelas 0 dan 2 (recall hanya sekitar 82–83%). Hal ini kemungkinan besar disebabkan oleh asumsi independensi antar fitur yang tidak sepenuhnya berlaku di data ini.

### Kesimpulan Akhir
- SVM dan Random Forest merupakan dua model terbaik dengan keseimbangan performa tinggi, stabil, dan generalisasi yang kuat.

- Decision Tree menunjukkan performa sempurna, tetapi perlu diwaspadai karena kemungkinan besar mengalami overfitting, meskipun data uji menghasilkan akurasi tinggi.

- KNN menunjukkan performa baik, namun sensitif terhadap distribusi data dan cenderung memiliki penurunan kecil pada recall.

- Gaussian Naive Bayes adalah model dengan performa paling rendah, tetapi masih cukup baik untuk baseline, terutama karena kesederhanaannya.
"""