# Topic Modeling Berita Detik.com
### Perbandingan LDA, NMF, LSA, dan BERTopic

> **Tugas Akhir / Penelitian NLP** — Implementasi topic modeling pada dataset berita berbahasa Indonesia hasil web scraping, dengan evaluasi menggunakan coherence score (c_v dan c_npmi).

---

## Daftar Isi

- [Gambaran Umum](#gambaran-umum)
- [Dataset](#dataset)
- [Metode](#metode)
- [Hasil Evaluasi](#hasil-evaluasi)
- [Analisis Topik per Model](#analisis-topik-per-model)
- [Topic Overlap Antar Model](#topic-overlap-antar-model)
- [Analisis Noise BERTopic](#analisis-noise-bertopic)
- [Stabilitas Model LDA](#stabilitas-model-lda)
- [Struktur File](#struktur-file)
- [Cara Menjalankan](#cara-menjalankan)
- [Referensi](#referensi)

---

## Gambaran Umum

Penelitian ini mengimplementasikan dan membandingkan empat algoritma topic modeling pada korpus berita berbahasa Indonesia dari Detik.com. Topic modeling digunakan untuk menemukan tema-tema tersembunyi dalam kumpulan dokumen teks tanpa supervisi (unsupervised).

**Tujuan penelitian:**
- Mengidentifikasi topik-topik dominan dalam berita Indonesia secara otomatis
- Membandingkan performa LDA, NMF, LSA, dan BERTopic secara kuantitatif (coherence score) dan kualitatif (interpretabilitas topik)
- Menganalisis konsistensi topik yang ditemukan lintas model

---

## Dataset

| Atribut | Detail |
|---|---|
| Sumber | Web scraping Detik.com |
| Total artikel | 7.164 dokumen |
| Jumlah kategori | 9 kategori |
| Rata-rata panjang | 191 kata per artikel (setelah stemming) |
| Rentang panjang | 20 — 3.766 kata |

**Distribusi kategori:**

| Kategori | Jumlah | Proporsi |
|---|---|---|
| Olahraga | 1.419 | 19,8% |
| Ekonomi | 1.374 | 19,2% |
| Gaya Hidup | 1.007 | 14,1% |
| Politik | 954 | 13,3% |
| Otomotif | 556 | 7,8% |
| Kesehatan | 497 | 6,9% |
| Kuliner | 487 | 6,8% |
| Travel | 473 | 6,6% |
| Hiburan | 397 | 5,5% |

**Kolom dataset:**

```
kategori | judul | isi | teks_clean | teks_norm | teks_nostop | teks_stem |
jumlah_kata | kata_clean | kata_nostop | kata_stem
```

Kolom `teks_stem` (hasil stemming Sastrawi) digunakan sebagai input utama semua model.

---

## Metode

### Pipeline Preprocessing

```
Raw Text → Cleaning → Normalisasi → Stopword Removal → Stemming (Sastrawi)
                                                              ↓
                                                        teks_stem
                                                              ↓
                              Gensim Dictionary (filter: min_df=5, max_df=0.70)
                                                              ↓
                                                    11.675 token unik
                                                    7.164 dokumen
```

### Model yang Digunakan

#### 1. LDA (Latent Dirichlet Allocation)
- **Paradigma:** Probabilistik Bayesian
- **Input:** Bag-of-Words (count matrix)
- **Pencarian k optimal:** k = 5 hingga 15 (evaluasi c_v + log-perplexity)
- **Konfigurasi terbaik:** k=15, passes=10, iterations=100, alpha=symmetric
- **Visualisasi:** pyLDAvis interaktif

#### 2. NMF (Non-negative Matrix Factorization)
- **Paradigma:** Aljabar (Matrix Factorization)
- **Input:** TF-IDF matrix (max_df=0.85, min_df=5, max_features=15.000)
- **Pencarian k optimal:** k = 5 hingga 15 (evaluasi c_v + reconstruction error)
- **Konfigurasi terbaik:** k=15, init=nndsvda, max_iter=500

#### 3. LSA (Latent Semantic Analysis)
- **Paradigma:** Aljabar (Singular Value Decomposition)
- **Input:** TF-IDF matrix
- **Pencarian k optimal:** k = 5 hingga 15 (evaluasi c_v)
- **Konfigurasi terbaik:** k=5 (coherence turun monoton seiring k naik)

#### 4. BERTopic
- **Paradigma:** Neural (Contextual Embedding)
- **Embedding model:** `paraphrase-multilingual-MiniLM-L12-v2` (384 dimensi)
- **Dimensionality reduction:** UMAP (n_neighbors=15, n_components=5)
- **Clustering:** HDBSCAN (min_cluster_size=30)
- **Jumlah topik:** otomatis — ditemukan 21 topik

---

## Hasil Evaluasi
<img width="1589" height="1181" alt="1output" src="https://github.com/user-attachments/assets/9be6a5e0-92e5-4b0d-a9fa-df27eef71ad5" />


### Tabel Perbandingan Coherence Score

| Model | Paradigma | Input | Best k | c_v | c_npmi | Outlier Handling |
|---|---|---|---|---|---|---|
| **NMF** | Aljabar (Matrix Factorization) | TF-IDF | 15 | **0.7871** | **0.2088** | Tidak |
| **BERTopic** | Neural (Contextual Embedding) | Sentence Embedding | 21 (auto) | 0.7647 | 0.1885 | Ya (2.222 dok, 31%) |
| **LDA** | Probabilistik (Bayesian) | Bag-of-Words | 15 | 0.5945 | 0.0845 | Tidak |
| **LSA** | Aljabar (SVD) | TF-IDF | 5 | 0.4850 | 0.0409 | Tidak |

### Interpretasi Hasil

**NMF unggul secara kuantitatif** dengan c_v = 0.7871 — tertinggi di antara keempat model. Sifat dekomposisi non-negatif TF-IDF menghasilkan topik yang sparse dan distinktif, sehingga kata-kata dalam satu topik benar-benar sering muncul bersama tanpa tumpang tindih. Topik-topik NMF sangat bersih: MotoGP, bulu tangkis, zodiak, kuliner, dan kesehatan masing-masing terpilah dengan jelas.

**BERTopic berada di posisi kedua** (c_v = 0.7647) dengan keunggulan unik: menemukan jumlah topik secara otomatis dan mampu mendeteksi dokumen outlier (topic -1). Meskipun angka coherence-nya sedikit di bawah NMF, BERTopic menghasilkan 21 topik yang lebih granular dibandingkan model lain dengan k=15.

**LDA menghasilkan coherence sedang** (c_v = 0.5945). Beberapa topik LDA menunjukkan tumpang tindih semantik — misalnya topik yang mencampur kata politik dengan kriminalitas, atau topik yang menggabungkan isu internasional dengan olahraga. Hal ini wajar karena LDA memodelkan setiap dokumen sebagai campuran topik (soft assignment).

**LSA menghasilkan coherence terendah** (c_v = 0.4850) dengan k optimal hanya 5 topik. SVD tidak membatasi nilai negatif sehingga representasi topik kurang interpretable. Topik LSA cenderung terlalu umum — Topic 1 mencampur hampir semua kata frekuensi tinggi dari berbagai domain.

### Pencarian k Optimal

```
LDA Coherence per k:
k= 5 → 0.4996    k= 9 → 0.5222    k=13 → 0.5413
k= 6 → 0.4869    k=10 → 0.5549    k=14 → 0.5247
k= 7 → 0.4853    k=11 → 0.5408    k=15 → 0.5945 ← best
k= 8 → 0.5048    k=12 → 0.5734

NMF Coherence per k:
k= 5 → 0.7163    k= 9 → 0.7108    k=13 → 0.7257
k= 6 → 0.7237    k=10 → 0.7078    k=14 → 0.7770
k= 7 → 0.7075    k=11 → 0.7294    k=15 → 0.7871 ← best
k= 8 → 0.6921    k=12 → 0.7545

LSA Coherence per k (turun monoton):
k= 5 → 0.4850 ← best    k=10 → 0.3965
k= 6 → 0.4647           k=11 → 0.3830
k= 7 → 0.4377           k=12 → 0.3759
k= 8 → 0.4255           k=13 → 0.3656
k= 9 → 0.4091           k=14 → 0.3610    k=15 → 0.3576
```

---

## Analisis Topik per Model

### LDA — Top Words per Topik (k=15)

| Topik | Label | Top 5 Kata |
|---|---|---|
| T1 | Hukum & Korupsi | hukum, tahun, dakwa, pasal, sehat |
| T2 | Politik & Kriminalitas | prabowo, laku, korban, polisi, lapor |
| T3 | MotoGP & Balap Motor | kapal, israel, negara, atlet, iran |
| T4 | Ekonomi & UMKM | harga, jual, produk, emas, juta |
| T5 | Pasar Saham & Keuangan | pasar, saham, ekonomi, ihsg, indeks |
| T6 | Otomotif & Kendaraan Listrik | mobil, kendara, listrik, motor, pajak |
| T7 | Kesehatan & Gaya Hidup | makan, sehat, tubuh, kulit, kopi |
| T8 | Bulu Tangkis | gim, menang, ganda, putri, alwi |
| T9 | Infrastruktur & Pemerintahan | jakarta, jalan, menteri, kapal, daerah |
| T10 | Olahraga & Atletik | atlet, main, juara, latih, olahraga |
| T11 | Hiburan & Zodiak | zodiak, ramal, harga, cinta, restoran |
| T12 | Selebriti & Media Sosial | anak, tahun, tampil, jalan, instagram |
| T13 | Travel & Kuliner | makan, restoran, menu, ayam, goreng |
| T14 | Otomotif & Balap | balap, moto, marc, marquez, posisi |
| T15 | Kesehatan & Konsumsi | sehat, tubuh, sakit, konsumsi, darah |

> Catatan: T3 LDA menggabungkan isu internasional dengan olahraga — contoh tumpang tindih topik yang menjadi limitasi LDA pada dataset multi-domain.

### NMF — Top Words per Topik (k=15)

| Topik | Label | Top 5 Kata |
|---|---|---|
| T1 | Politik & Ekspor SDA | prabowo, ekonomi, purbaya, ekspor, perintah |
| T2 | MotoGP & Balap Motor | balap, moto, marquez, marc, alex |
| T3 | Zodiak & Ramalan | zodiak, ramal, asmara, untung, cinta |
| T4 | Bulu Tangkis | gim, main, menang, ganda, putri |
| T5 | Pasar Saham & IHSG | saham, ihsg, lemah, dagang, indeks |
| T6 | Kuliner & Restoran | makan, restoran, menu, ayam, goreng |
| T7 | Berita Internasional & WNI | kapal, israel, wni, manusia, gpci |
| T8 | Otomotif & Kendaraan Listrik | kendara, listrik, mobil, motor, pajak |
| T9 | Travel & Event | jogja, run, lari, festival, financial |
| T10 | Perbankan & Diskon | harga, bank, diskon, mega, transmart |
| T11 | Hiburan & Selebriti | instagram, tampil, anak, dok, film |
| T12 | Balap Motor Lokal | veda, balap, ega, posisi, finis |
| T13 | Kriminalitas & Kereta | korban, kereta, polisi, laku, lintas |
| T14 | Kesehatan & Tubuh | sehat, tubuh, sakit, konsumsi, darah |
| T15 | Olahraga & Atletik | atlet, olahraga, games, latih, asi |

### LSA — Top Words per Topik (k=5)

| Topik | Label | Top 5 Kata |
|---|---|---|
| T1 | Umum / Campuran | balap, tahun, jalan, harga, uang |
| T2 | MotoGP & Balap Motor | balap, moto, marquez, posisi, marc |
| T3 | Zodiak & Hiburan | zodiak, ramal, asmara, untung, cinta |
| T4 | Olahraga & Atletik | saham, main, putri, atlet, juara |
| T5 | Ekonomi & Pemerintahan | ekonomi, perintah, negara, prabowo, tumbuh |

> LSA hanya mampu memisahkan 5 topik besar karena SVD tidak mempertahankan nilai non-negatif, menghasilkan representasi yang lebih difus.

### BERTopic — 21 Topik Otomatis

| Topik | Count | Label |
|---|---|---|
| T0 | 1.895 | Kuliner, Ekonomi & Politik Campuran |
| T1 | 656 | Hiburan & Selebriti |
| T2 | 512 | MotoGP & Balap Motor |
| T3 | 505 | Bulu Tangkis |
| T4 | 364 | Otomotif & Kendaraan Listrik |
| T5 | 131 | Perbankan & Nilai Tukar |
| T6 | 122 | Zodiak & Ramalan Asmara |
| T7 | 101 | Geopolitik & Konflik Global |
| T8 | 85 | Kesehatan & Isu Global (WHO) |
| T9 | 76 | Travel & Hotel |
| T10 | 75 | Event Lari & Olahraga |
| T11 | 74 | Konflik Israel & WNI Luar Negeri |
| T12 | 54 | Pelatihan Atlet Bulu Tangkis |
| T13 | 48 | Sepak Bola & Timnas |
| T14 | 43 | Penerbangan & Maskapai |
| T15 | 39 | Kriminalitas & Penganiayaan |
| T16 | 34 | Festival Keuangan & Ekonomi |
| T17 | 33 | Agama & Sertifikasi Islam |
| T18 | 32 | Perawatan Kulit & Kecantikan |
| T19 | 32 | Ramalan Zodiak & Keberuntungan |
| T20 | 31 | Promo Diskon & Kartu Bank |
| -1 | 2.222 | Noise (multi-tema / outlier) |

> BERTopic memisahkan bulu tangkis menjadi dua topik granular: T3 (pertandingan/hasil) dan T12 (pelatihan/pembinaan atlet) — perbedaan yang tidak dapat ditangkap model lain.

---

## Topic Overlap Antar Model

### Cosine Similarity Antar Model (rata-rata top words)

| | LDA | NMF | LSA | BERTopic |
|---|---|---|---|---|
| **LDA** | 1.000 | 0.456 | 0.424 | 0.266 |
| **NMF** | 0.456 | 1.000 | 0.491 | 0.479 |
| **LSA** | 0.424 | 0.491 | 1.000 | 0.289 |
| **BERTopic** | 0.266 | 0.479 | 0.289 | 1.000 |

**Interpretasi:**
- NMF dan LSA memiliki similarity tertinggi (0.491) — wajar karena keduanya menggunakan TF-IDF sebagai input dan pendekatan aljabar
- BERTopic paling berbeda dari LDA (0.266) — mencerminkan perbedaan fundamental antara pendekatan bag-of-words vs. contextual embedding
- NMF–BERTopic (0.479) memiliki similarity cukup tinggi meskipun paradigma berbeda, mengindikasikan keduanya menemukan topik yang semantically similar

### Topik yang Konsisten di Semua Model

| Tema | LDA | NMF | LSA | BERTopic | Konsistensi |
|---|---|---|---|---|---|
| MotoGP & Balap Motor | T3 | T2 | T2 | T2 | Semua model |
| Bulu Tangkis | T8 | T4 | — | T3 | LDA, NMF, BERTopic |
| Pasar Saham & Keuangan | T5 | T5 | — | T5 | LDA, NMF, BERTopic |
| Otomotif & Kendaraan Listrik | T6 | T8 | — | T4 | LDA, NMF, BERTopic |
| Kesehatan & Gaya Hidup | T7/T15 | T14 | — | T8 | LDA, NMF, BERTopic |
| Hiburan & Selebriti | T11 | T11 | T3 | T1 | Semua model |
| Zodiak & Ramalan | T11 | T3 | T3 | T6 | Semua model |
| Kuliner & Restoran | T13 | T6 | — | T10 | LDA, NMF, BERTopic |
| Politik & Pemerintahan | T2/T9 | T1 | T5 | T8 | Semua model |
| Olahraga & Atletik | T10 | T15 | T4 | T10 | Semua model |

Topik **MotoGP, Zodiak, Hiburan, Politik, dan Olahraga** konsisten muncul di semua model — mengonfirmasi bahwa topik-topik tersebut memang merupakan tema dominan yang kuat dalam dataset berita Detik.com.

---

## Analisis Noise BERTopic

### Distribusi Dokumen Noise per Kategori

| Kategori | Total Artikel | Noise Docs | Noise % |
|---|---|---|---|
| **Travel** | 473 | 230 | **48.6%** |
| **Ekonomi** | 1.374 | 657 | **47.8%** |
| **Politik** | 954 | 444 | **46.5%** |
| Kesehatan | 497 | ~220 | ~44% |
| Gaya Hidup | 1.007 | ~380 | ~38% |
| Kuliner | 487 | ~160 | ~33% |
| Hiburan | 397 | ~100 | ~25% |
| Otomotif | 556 | ~90 | ~16% |
| Olahraga | 1.419 | ~80 | ~6% |

**Total noise: 2.222 dokumen (31.0%)** dari 7.164 artikel

**Insight kunci:**

Travel memiliki noise tertinggi (48.6%) karena artikel travel di Detik.com sering menggabungkan informasi destinasi wisata, kuliner lokal, transportasi, akomodasi, dan tips perjalanan dalam satu artikel — sehingga HDBSCAN tidak dapat mengklasifikasikannya ke satu topik tunggal.

Ekonomi dan Politik (masing-masing ~47%) berada di posisi kedua dan ketiga karena berita ekonomi Indonesia sering menyentuh kebijakan pemerintah (politik), sedangkan berita politik sering membahas implikasi ekonomi — terjadi cross-domain overlap yang tinggi.

Olahraga memiliki noise terendah (hanya ~6%) karena artikel olahraga cenderung sangat spesifik pada satu cabang olahraga tertentu (MotoGP, bulu tangkis, sepak bola), sehingga mudah dikelompokkan.

**Implikasi:** 31% noise menunjukkan bahwa hampir sepertiga artikel Detik.com bersifat multi-tema — informasi ini tidak dapat diperoleh dari LDA maupun NMF yang memaksakan setiap dokumen ke minimal satu topik.

---

## Stabilitas Model LDA

LDA bersifat stokastik (bergantung pada random seed), sehingga hasil dapat bervariasi antar run. Uji stabilitas dilakukan dengan 3 random seed berbeda:

| Seed | c_v | c_npmi | Waktu |
|---|---|---|---|
| 42 | 0.5945 | 0.0845 | 31 detik |
| 0 | 0.5831 | 0.0820 | 31 detik |
| 123 | 0.5311 | 0.0573 | 33 detik |

**Statistik variance:**
- c_v: mean = 0.5696 | std = **0.0338** | range = 0.0634
- c_npmi: mean = 0.0746 | std = **0.0150** | range = 0.0272

**Kesimpulan:** LDA menunjukkan **ketidakstabilan moderat** — range c_v sebesar 0.0634 antar seed cukup signifikan. Ini berarti hasil LDA (coherence = 0.5945 dengan seed=42) adalah hasil terbaik yang mungkin diperoleh, bukan nilai yang konsisten. Untuk penelitian yang membutuhkan reprodusibilitas, NMF atau BERTopic lebih disarankan karena deterministik (NMF) atau memiliki variance lebih rendah pada embedding yang sama (BERTopic).



## Referensi

1. **IEEE Xplore (2025)** — *Topic Modelling Analysis on Indonesian News Using BERT Topic Model*. DOI: [10.1109/ICAICTA64568.2024.10903779](https://ieeexplore.ieee.org/document/10903779/)

2. **ScienceDirect / Information Systems (2022)** — *Topic Modeling Algorithms and Applications: A Survey*. DOI: [10.1016/j.is.2022.101869](https://www.sciencedirect.com/science/article/abs/pii/S0306437922001090)

3. **IEEE Access (2022)** — *Topic Modeling: Perspectives From a Literature Review*. Robledo & Zuluaga. DOI: [10.1109/ACCESS.2022.3232939](https://ieeexplore.ieee.org/document/10002352/)

4. **ScienceDirect / Neurocomputing (2025)** — *Enhancing Topic Coherence and Diversity in Document Embeddings Using LLMs: A Focus on BERTopic*. DOI: [10.1016/j.neucom.2025.129638](https://www.sciencedirect.com/science/article/abs/pii/S095741742501139X)

5. **Blei, D., Ng, A., & Jordan, M. (2003)** — *Latent Dirichlet Allocation*. Journal of Machine Learning Research, 3, 993—1022.

6. **Grootendorst, M. (2022)** — *BERTopic: Neural topic modeling with a class-based TF-IDF procedure*. arXiv:2203.05794.

7. **Lee, D., & Seung, H. S. (1999)** — *Learning the parts of objects by non-negative matrix factorization*. Nature, 401, 788—791.

---

*Dataset scraping Detik.com — 9 kategori berita — 7.164 artikel*
