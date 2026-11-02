# Lampiran B — Glosarium

Istilah diurutkan menurut abjad berdasarkan istilah bahasa Inggrisnya. Bab tempat istilah diperkenalkan ada dalam tanda kurung siku.

| Istilah | Arti |
|---|---|
| **A/B test** (*uji A/B*) | Eksperimen daring yang membagi pengguna secara acak antara kelompok kontrol dan perlakuan lalu membandingkan sebuah metrik; uji signifikansi menentukan apakah perbedaannya nyata. [14] |
| **Ablation** (*ablasi*) | Melatih ulang model dengan satu bagian dihilangkan (di sini: satu kelompok fitur) untuk mengukur seberapa besar kontribusi bagian itu. [10] |
| **Add-to-cart** (*tambah ke keranjang*) | Event yang dicatat saat pengguna memasukkan item ke keranjang; sinyal minat yang lebih kuat daripada klik. [1] |
| **ALS (Alternating Least Squares)** (*kuadrat terkecil bergantian*) | Algoritma faktorisasi matriks yang menetapkan vektor item lalu menyelesaikan vektor pengguna secara eksak, kemudian sebaliknya, berulang-ulang. [6] |
| **ANN (Approximate Nearest Neighbour)** (*tetangga terdekat hampiran*) | Pencarian yang menemukan vektor yang *hampir* paling dekat jauh lebih cepat daripada memeriksa setiap vektor. [7] |
| **AP (Average Precision)** (*presisi rata-rata*) | Rata-rata precision@k yang diambil pada setiap peringkat k di mana item relevan muncul. [2] |
| **Bayesian average** (*rata-rata Bayesian*) | Rating atau rasio yang ditarik ke arah prior global sampai data cukup terkumpul, sehingga 1 klik dari 1 tayangan tidak dianggap rasio 100%. [3] |
| **BPR (Bayesian Personalised Ranking)** (*pemeringkatan personal Bayesian*) | Fungsi tujuan berpasangan (*pairwise*): untuk setiap pengguna, beri skor item yang pernah ia interaksikan lebih tinggi daripada item yang belum. [6] |
| **Candidate** (*kandidat*) | Item yang diusulkan oleh sebuah retriever untuk seorang pengguna; hanya kandidat yang sampai ke ranker. [8] |
| **Candidate recall** (*recall kandidat*) | Porsi item positif pengguna di masa depan yang ada dalam himpunan kandidat; batas atas bagi ranker mana pun. [8] |
| **Co-visitation** (*ko-kunjungan*) | Menghitung seberapa sering dua item diinteraksikan dalam sesi yang sama; dasar dari "dilihat bersamaan". [5] |
| **Cold start** (*awal dingin*) | Pengguna atau item baru dengan sedikit atau tanpa riwayat, sehingga metode personal tidak punya bahan untuk dipelajari. [12] |
| **Collaborative filtering** (*penyaringan kolaboratif*) | Merekomendasikan berdasarkan perilaku pengguna yang mirip atau konsumsi bersama antar item, tanpa melihat konten item. [5] |
| **Content-based filtering** (*penyaringan berbasis konten*) | Merekomendasikan item yang atributnya (teks, kategori, merek) mirip dengan item yang disukai pengguna. [4] |
| **Cosine similarity** (*kemiripan kosinus*) | Kosinus sudut antara dua vektor: 1 untuk arah yang sama, 0 untuk yang tidak berkaitan. [4] |
| **Coverage** (*cakupan*) | Porsi katalog yang muncul di top-K siapa pun; ukuran keberagaman sederhana. [2] |
| **CTR (Click-through rate)** (*rasio klik*) | Jumlah klik dibagi jumlah impresi. [1] |
| **Cutoff** (*titik potong waktu*) | Saat di mana fitur dihitung; hanya event sebelumnya yang boleh dipakai. [2] |
| **DCG / NDCG** | Discounted cumulative gain: relevansi dijumlahkan dengan diskon $1/\log_2(\text{rank}+1)$; NDCG membaginya dengan DCG terbaik yang mungkin, sehingga bernilai 0–1. [2] |
| **Decay (half-life)** (*peluruhan / waktu paruh*) | Membobot event lama secara eksponensial lebih kecil; dengan waktu paruh 7 hari, aksi seminggu lalu dihitung setengah. [3] |
| **Dwell time** (*durasi tinggal*) | Detik yang dihabiskan pengguna di halaman produk; lebih lama biasanya berarti lebih berminat. [1] |
| **Embedding** (*representasi vektor*) | Vektor yang dipelajari untuk merepresentasikan pengguna atau item sehingga hal-hal yang mirip saling berdekatan. [7] |
| **Event** (*kejadian*) | Satu aksi pengguna yang dicatat pada sebuah item: impresi, klik, add-to-cart, order, hide, atau return. [1] |
| **Explicit feedback** (*umpan balik eksplisit*) | Rating atau ulasan yang sengaja diberikan pengguna. [1] |
| **Exploration** (*eksplorasi*) | Sengaja menampilkan item yang belum pasti untuk mempelajarinya, dengan biaya jangka pendek yang kecil (ε-greedy, Thompson sampling). [11] |
| **Fallback** (*cadangan*) | Apa yang disajikan sistem saat recommender tidak tersedia atau pengguna tidak dikenal; di sini, produk terlaris terbaru. [12] |
| **Feature** (*fitur*) | Angka yang menggambarkan pasangan (pengguna, item) pada suatu saat, dipakai sebagai masukan model. [9] |
| **Feature store** (*penyimpanan fitur*) | Sistem yang menghitung, menyimpan, dan menyajikan fitur secara konsisten untuk pelatihan dan penyajian. [14] |
| **Feedback loop** (*putaran umpan balik*) | Rekomendasi model membentuk data yang kelak dipakai melatihnya, sehingga dapat memperkuat biasnya sendiri. [12] |
| **Gradient boosting** | Membangun model sebagai jumlah pohon keputusan kecil, masing-masing dilatih untuk memperbaiki kesalahan pohon sebelumnya. [10] |
| **Hard filter** (*filter keras*) | Aturan yang langsung membuang kandidat (stok habis, disembunyikan pengguna). [8] |
| **Impression** (*impresi*) | Sebuah item ditampilkan di layar. [1] |
| **Implicit feedback** (*umpan balik implisit*) | Perilaku yang menandakan minat tanpa berupa rating: melihat, klik, membeli. [1] |
| **In-batch negatives** (*negatif dalam batch*) | Memakai positif lain dalam satu batch pelatihan sebagai negatif bagi setiap contoh. [7] |
| **IPW (Inverse propensity weighting)** (*pembobotan propensitas terbalik*) | Membobot ulang umpan balik yang tercatat dengan kebalikan probabilitas sebuah item dilihat, untuk mengoreksi bias eksposur dan posisi. [12] |
| **Item-to-item** (*item ke item*) | Collaborative filtering yang merekomendasikan item mirip dengan item yang diinteraksikan pengguna, memakai kemiripan item–item. [5] |
| **IVF (Inverted file index)** (*indeks berkas terbalik*) | Metode ANN yang mengelompokkan vektor dan hanya mencari di klaster terdekat dengan kueri. [7] |
| **Label** | Apa yang sebenarnya terjadi setelah cutoff (0 tidak ada, 1 klik, 2 keranjang, 3 order); target pelatihan model. [9] |
| **LambdaMART** | Ranker pohon gradient-boosted yang gradiennya dibobot oleh seberapa besar pertukaran dua item akan mengubah NDCG. [10] |
| **Latency budget** (*anggaran latensi*) | Waktu yang boleh dipakai tiap tahap sebuah permintaan agar seluruh respons memenuhi tenggatnya. [14] |
| **Leakage** (*kebocoran data*) | Memakai informasi dari setelah cutoff saat membangun fitur; menggelembungkan metrik offline. [9] |
| **Learning to rank** (*belajar memeringkat*) | Melatih model untuk mengurutkan item bagi suatu kueri atau pengguna, bukan memprediksi nilai tiap item secara terpisah. [10] |
| **logQ correction** (*koreksi logQ*) | Mengurangi logit sebuah item dengan log probabilitas pengambilan sampelnya saat berlatih dengan in-batch negatives, agar item populer tidak dihukum berlebihan. [7] |
| **Long tail** (*ekor panjang*) | Banyak item yang masing-masing mendapat sedikit perhatian tetapi bersama-sama membentuk porsi besar katalog. [1] |
| **Matrix factorisation** (*faktorisasi matriks*) | Menghampiri matriks pengguna–item sebagai hasil kali matriks faktor pengguna dan item berdimensi rendah. [6] |
| **MMR (Maximal Marginal Relevance)** (*relevansi marginal maksimal*) | Aturan re-ranking yang menimbang relevansi terhadap kemiripan dengan item yang sudah dipilih. [11] |
| **Multi-task learning** (*pembelajaran multi-tugas*) | Satu model memprediksi beberapa hasil (klik, keranjang, order) dengan lapisan bersama. [14] |
| **Point-in-time correctness** (*kebenaran pada titik waktu*) | Menghitung setiap fitur hanya dari data yang tersedia pada cutoff-nya. [9] |
| **Popularity bias** (*bias popularitas*) | Recommender cenderung memihak item yang sudah populer, yang lalu mendapat eksposur lebih banyak dan makin populer. [3] |
| **Position bias** (*bias posisi*) | Item yang tampil lebih tinggi mendapat lebih banyak klik terlepas dari relevansinya. [12] |
| **Precision@K / Recall@K** | Porsi top K yang relevan / porsi item relevan yang berada di top K. [2] |
| **Re-ranking** (*pemeringkatan ulang*) | Tahap akhir yang menerapkan aturan tingkat halaman (batas keberagaman, filter sudah-dibeli) pada skor model. [11] |
| **Retrieval** (*pengambilan*) | Tahap pertama: memilih secara murah beberapa ratus kandidat dari seluruh katalog. [8] |
| **Session** (*sesi*) | Rentetan aktivitas pengguna; di sini jeda 30 menit memulai sesi baru. [13] |
| **SHAP** | Metode yang memecah satu prediksi menjadi kontribusi per fitur yang jumlahnya sama dengan keluaran model. [10] |
| **Smoothing** (*penghalusan*) | Lihat *Bayesian average*. [3] |
| **Sparsity** (*kejarangan*) | Porsi matriks pengguna–item yang kosong; biasanya di atas 99%. [1] |
| **TF-IDF** | Term frequency × inverse document frequency: bobot kata yang tinggi untuk kata yang sering muncul dalam satu dokumen tetapi jarang secara keseluruhan. [4] |
| **Training/serving skew** (*ketidaksesuaian pelatihan/penyajian*) | Perbedaan cara fitur dihitung saat pelatihan dan saat penyajian, yang diam-diam menurunkan kualitas model. [14] |
| **Two-tower model** (*model dua menara*) | Model retrieval dengan jaringan saraf terpisah untuk pengguna dan item yang vektor keluarannya dibandingkan dengan dot product. [7] |
