# Lampiran B — Glosarium

Istilah diurutkan menurut abjad berdasarkan istilah bahasa Inggrisnya. Setiap entri memberi arti dalam bahasa sederhana, kadang dengan contoh kecil. Bab tempat istilah diperkenalkan dan dijelaskan lengkap ada dalam tanda kurung siku.

| Istilah | Arti |
|---|---|
| **A/B test** (*uji A/B*) | Eksperimen di mana separuh pembeli yang dipilih acak melihat sistem lama (kontrol) dan separuh lainnya melihat sistem baru (perlakuan), lalu uji statistik memberi tahu apakah selisih sebuah metrik itu nyata atau hanya kebetulan. [14] |
| **Ablation** (*ablasi*) | Melatih ulang model dengan satu kelompok fitur dihilangkan untuk melihat seberapa buruk hasilnya; penurunan itu menunjukkan seberapa berharga kelompok tersebut. [10] |
| **Add-to-cart** (*tambah ke keranjang*) | Event yang dicatat saat pembeli memasukkan item ke keranjang; menandakan minat yang lebih kuat daripada klik. [1] |
| **ALS (Alternating Least Squares)** (*kuadrat terkecil bergantian*) | Cara mempelajari kartu selera (lihat *Matrix factorisation*): tetapkan kartu item lalu cari kartu pengguna terbaik, kemudian tetapkan kartu pengguna dan cari kartu item, dan ulangi. [6] |
| **ANN (Approximate Nearest Neighbour)** (*tetangga terdekat hampiran*) | Pencarian yang menemukan item yang *hampir* paling dekat dengan kueri, jauh lebih cepat daripada membandingkan dengan setiap item. [7] |
| **AP (Average Precision)** (*presisi rata-rata*) | Skor untuk satu daftar: di setiap peringkat tempat item relevan muncul, catat porsi item relevan sejauh itu, lalu rata-ratakan porsi tersebut. [2] |
| **Bayesian average** (*rata-rata Bayesian*) | Rasio yang ditarik ke arah rata-rata keseluruhan sampai sebuah item punya cukup data, sehingga 1 klik dari 1 tayangan tidak dianggap sempurna 100%. Disebut juga *smoothing*. [3] |
| **Beacon** (*suar*) | Fitur browser yang menyerahkan pesan kecil kepada browser untuk dikirim setelah halaman ditutup, seperti memasukkan surat ke kotak pos saat hendak pergi. [13] |
| **Beta distribution** (*distribusi Beta*) | Kurva yang menunjukkan seberapa masuk akal tiap nilai rasio klik sebuah item, berdasarkan jumlah klik dan non-klik sejauh ini; sempit bila datanya banyak dan lebar bila datanya sedikit. [11] |
| **BPR (Bayesian Personalised Ranking)** (*pemeringkatan personal Bayesian*) | Tujuan pelatihan yang berbunyi: untuk setiap pembeli, beri skor lebih tinggi pada item yang pernah ia interaksikan daripada item yang belum. [6] |
| **Candidate** (*kandidat*) | Item yang diusulkan retriever tahap pertama untuk seorang pembeli; hanya kandidat yang diberi skor oleh ranker. [8] |
| **Candidate recall** (*recall kandidat*) | Porsi item yang kelak benar-benar diminati pembeli yang berhasil masuk ke daftar kandidat. Inilah batas atas recall: ranker tidak bisa merekomendasikan apa yang tidak pernah diusulkan. [8] |
| **Co-visitation** (*ko-kunjungan*) | Menghitung seberapa sering dua item dipakai dalam sesi yang sama; mendasari "orang yang melihat ini juga melihat itu". [5] |
| **Cold start** (*awal dingin*) | Pembeli atau item baru dengan sedikit atau tanpa riwayat, sehingga belum ada bahan bagi metode personal untuk belajar. [12] |
| **Collaborative filtering** (*penyaringan kolaboratif*) | Merekomendasikan dari perilaku banyak pembeli (pembeli yang mirip, atau item yang dipakai bersama), tanpa membaca isi item. [5] |
| **Content-based filtering** (*penyaringan berbasis konten*) | Merekomendasikan item yang deskripsinya (teks, kategori, merek) mirip dengan item yang sudah disukai pembeli. [4] |
| **Cosine similarity** (*kemiripan kosinus*) | Skor 0 sampai 1 untuk seberapa mirip dua vektor, berdasarkan sudut di antara keduanya: 1 berarti arah yang sama, 0 berarti tidak ada kesamaan. [4] |
| **Coverage** (*cakupan*) | Porsi katalog yang muncul di top-K siapa pun; cakupan rendah berarti banyak item tidak pernah ditampilkan. [2] |
| **CSR (Compressed Sparse Row)** (*baris jarang terkompresi*) | Cara menyimpan tabel yang hampir kosong dengan hanya menyimpan sel yang terisi beserta letaknya, baris demi baris. [5] |
| **CTR (Click-through rate)** (*rasio klik*) | Klik dibagi impresi: dari 100 tayangan, 5 klik berarti CTR 5%. [1] |
| **Cross feature** (*fitur silang*) | Fitur tentang sebuah pasangan, misalnya *pembeli ini × item ini*, bukan tentang pembeli atau item saja. [9] |
| **Cutoff** (*titik potong waktu*) | Saat di mana kita membekukan waktu: fitur hanya boleh memakai event sebelumnya, dan apa yang terjadi sesudahnya menjadi kunci jawaban. [2] |
| **DCG / NDCG** | DCG memberi nilai untuk tiap item relevan, makin kecil bila posisinya makin bawah ($1/\log_2(\text{rank}+1)$); NDCG membaginya dengan skor terbaik yang mungkin, sehingga bernilai 0 sampai 1. [2] |
| **Decay (half-life)** (*peluruhan / waktu paruh*) | Menghitung event yang lebih lama dengan bobot lebih kecil. Dengan waktu paruh 7 hari, aksi seminggu lalu dihitung setengah dari aksi hari ini. [3] |
| **Dense** (*padat*) | Disimpan penuh, setiap sel dicatat walau nilainya nol; kebalikan dari *sparse*. [6] |
| **Dot product** (*hasil kali titik*) | Kalikan dua daftar angka posisi demi posisi, lalu jumlahkan hasilnya: $(1,2)\cdot(3,4) = 1\times3 + 2\times4 = 11$. Hasil yang lebih besar berarti kedua daftar menunjuk arah yang sama. [4] |
| **Dwell time** (*durasi tinggal*) | Berapa detik pembeli bertahan di halaman produk; lebih lama biasanya berarti lebih berminat. [1] |
| **Embedding** (*representasi vektor*) | Daftar angka pendek yang dipelajari untuk seorang pembeli atau sebuah item, diatur sehingga hal yang mirip mendapat daftar yang mirip. [7] |
| **Epoch** | Satu putaran penuh melewati seluruh data pelatihan. [6] |
| **Event** (*kejadian*) | Satu aksi pembeli pada sebuah item yang dicatat: impresi, klik, add-to-cart, order, hide, atau return. [1] |
| **Explicit feedback** (*umpan balik eksplisit*) | Rating atau ulasan yang sengaja diberikan pembeli. [1] |
| **Exploration** (*eksplorasi*) | Sengaja menampilkan item yang belum pasti agar kita bisa mempelajarinya, dengan biaya jangka pendek yang kecil (ε-greedy dan Thompson sampling adalah dua caranya). [11] |
| **Exposure** (*eksposur*) | Apakah, dan seberapa sering, sebuah item benar-benar ditampilkan kepada pembeli; item yang tidak pernah ditampilkan tidak bisa mendapat klik. [1] |
| **Fallback** (*cadangan*) | Apa yang ditampilkan sistem saat recommender tidak tersedia atau pembeli tidak dikenal; di sini, produk terlaris terbaru. [12] |
| **Feature** (*fitur*) | Angka yang menggambarkan pasangan pembeli–item pada satu saat, seperti `user_clicks_30d`, yang dibaca model sebagai masukan. [9] |
| **Feature store** (*penyimpanan fitur*) | Tempat bersama yang mendefinisikan setiap fitur sekali saja dan menyajikan angka yang sama untuk pelatihan dan permintaan langsung. [14] |
| **Feedback loop** (*putaran umpan balik*) | Recommender menentukan apa yang ditampilkan, tampilan menentukan klik, dan klik menjadi data latih berikutnya, sehingga ia bisa memperkuat kebiasaannya sendiri. [12] |
| **Gradient** (*gradien*) | Untuk setiap angka yang boleh diubah model, sinyal yang menunjukkan arah (naik atau turun) yang membuat loss mengecil, dan seberapa besar. [6] |
| **Gradient boosting** | Membangun model dari banyak pohon keputusan kecil, masing-masing dilatih untuk memperbaiki kesalahan yang ditinggalkan pohon-pohon sebelumnya. [10] |
| **Guardrail metric** (*metrik pagar pengaman*) | Angka yang tidak boleh memburuk (misalnya cakupan atau kecepatan halaman) walau kita berusaha memperbaiki metrik utama. [2] |
| **Hard filter** (*filter keras*) | Aturan yang langsung membuang kandidat, seperti stok habis atau disembunyikan pembeli. [8] |
| **Hit rate@K** | Porsi pembeli yang setidaknya satu dari K item teratasnya adalah hit; bila 4 dari 10 pembeli menemukan sesuatu, nilainya 0.4. [2] |
| **HNSW (Hierarchical Navigable Small World)** | Metode ANN yang menghubungkan tiap vektor dengan tetangganya dalam graf berlapis dan menemukan kecocokan dengan menelusuri tautan. [7] |
| **Hyperparameter** (*hiperparameter*) | Pengaturan yang kamu pilih sebelum pelatihan (seperti ukuran embedding atau panjang jendela dalam hari), berbeda dari parameter yang dipelajari saat pelatihan. [3] |
| **Impression** (*impresi*) | Saat sebuah item ditampilkan di layar. [1] |
| **Implicit feedback** (*umpan balik implisit*) | Perilaku yang mengisyaratkan minat tanpa berupa rating: melihat, klik, membeli. [1] |
| **In-batch negatives** (*negatif dalam batch*) | Saat berlatih dengan satu batch contoh, memperlakukan item positif milik pembeli lain di batch itu sebagai "jawaban salah" bagi tiap pembeli. [7] |
| **IPW (Inverse propensity weighting)** (*pembobotan propensitas terbalik*) | Memberi bobot lebih pada klik tercatat yang peluang terlihatnya kecil (bobot = 1 ÷ peluang ditampilkan), untuk mengoreksi bias posisi dan eksposur. [12] |
| **Item-to-item** (*item ke item*) | Merekomendasikan item yang mirip dengan item yang dipakai pembeli, memakai tabel seberapa mirip pasangan item. [5] |
| **IVF (Inverted file index)** (*indeks berkas terbalik*) | Metode ANN yang mengelompokkan vektor ke beberapa grup dan hanya mencari di beberapa grup terdekat dengan kueri. [7] |
| **k-means** | Metode yang mengelompokkan titik ke $k$ grup: pilih $k$ pusat, tetapkan tiap titik ke pusat terdekat, pindahkan tiap pusat ke rata-rata titiknya, lalu ulangi. [7] |
| **Label** | Apa yang sebenarnya terjadi setelah cutoff (0 tidak ada, 1 klik, 2 keranjang, 3 order); jawaban yang dipelajari model untuk diprediksi. [9] |
| **Label window** (*jendela label*) | Rentang waktu tepat setelah cutoff (di sini satu minggu) yang event-nya menjadi label; tidak boleh dipakai untuk pelatihan atau fitur. [2] |
| **LambdaMART** | Ranker gradient-boosted yang sinyal pelatihannya lebih besar untuk pasangan item yang bila ditukar akan banyak mengubah NDCG. [10] |
| **Latency, p50, p99** (*latensi*) | Latensi adalah berapa lama satu permintaan berlangsung. p50 adalah waktu tipikal (median); p99 adalah waktu yang dikalahkan 99 dari setiap 100 permintaan, sehingga memperlihatkan yang lambat. [13] |
| **Latency budget** (*anggaran latensi*) | Waktu yang boleh dipakai tiap tahap sebuah permintaan agar seluruh respons tiba sebelum tenggatnya. [14] |
| **Latent factor** (*faktor laten*) | Dimensi selera tersembunyi (seperti "kuat dan hitam" versus "manis dan berkrim") yang tidak diberi label siapa pun tetapi ditemukan model dari data. [6] |
| **Leakage** (*kebocoran data*) | Memakai informasi dari setelah cutoff saat membangun fitur; ini curang dan menggelembungkan skor offline. [9] |
| **Learning rate** (*laju belajar*) | Seberapa besar langkah model setiap kali menyesuaikan angkanya; terlalu besar melampaui sasaran, terlalu kecil lambat. [6] |
| **Learning to rank** (*belajar memeringkat*) | Melatih model untuk menaruh item dalam urutan yang tepat bagi seorang pembeli, bukan memprediksi nilai tiap item secara terpisah. [10] |
| **Logit** | Skor mentah sebelum diubah menjadi probabilitas. [7] |
| **logQ correction** (*koreksi logQ*) | Mengurangi skor sebuah item dengan log seberapa sering item itu terambil sebagai negatif, agar item populer tidak dihukum tidak adil saat berlatih dengan in-batch negatives. [7] |
| **Long tail** (*ekor panjang*) | Banyak item yang masing-masing mendapat sedikit perhatian tetapi bersama-sama membentuk porsi besar katalog. [1] |
| **Loss** (*kerugian / galat*) | Satu angka untuk "seberapa salah model saat ini"; pelatihan berusaha mengecilkannya. Jika model memprediksi 3.5 dan kenyataannya 5, galat di balik loss-nya adalah 1.5. [6] |
| **Matrix factorisation** (*faktorisasi matriks*) | Meringkas tabel besar pembeli × item menjadi dua tabel tipis berisi "kartu selera" pendek, satu per pembeli dan satu per item, yang dot product-nya menebak sel yang kosong. [6] |
| **MMR (Maximal Marginal Relevance)** (*relevansi marginal maksimal*) | Aturan re-ranking yang memilih item berikutnya dengan menimbang seberapa relevan ia terhadap seberapa mirip ia dengan item yang sudah dipilih. [11] |
| **MRR (Mean Reciprocal Rank)** (*rata-rata peringkat kebalikan*) | Untuk tiap pembeli ambil 1 ÷ peringkat hit pertamanya (peringkat 1 bernilai 1, peringkat 2 bernilai 0.5, peringkat 4 bernilai 0.25), lalu rata-ratakan untuk semua pembeli. [2] |
| **Multi-task learning** (*pembelajaran multi-tugas*) | Satu model yang memprediksi beberapa hasil (klik, keranjang, order) sekaligus dengan berbagi lapisan dalamnya. [14] |
| **Overfitting** (*overfit*) | Model menghafal data latihnya, seperti siswa yang menghafal jawaban ujian tahun lalu lalu gagal di ujian tahun ini. [6] |
| **PCA (Principal Component Analysis)** (*analisis komponen utama*) | Cara memadatkan vektor berangka banyak menjadi dua angka dengan mempertahankan arah di mana titik-titik paling menyebar, sehingga bisa digambar di halaman. [7] |
| **Point-in-time correctness** (*kebenaran pada titik waktu*) | Menghitung setiap fitur hanya dari data yang ada pada cutoff-nya, seolah kita berdiri di saat itu. [9] |
| **Popularity bias** (*bias popularitas*) | Recommender cenderung memihak item yang sudah populer, yang lalu mendapat eksposur lebih banyak dan makin populer. [3] |
| **Position bias** (*bias posisi*) | Item yang tampil lebih tinggi di halaman mendapat lebih banyak klik walau tidak lebih baik. [12] |
| **PQ (Product quantisation)** (*kuantisasi produk*) | Memampatkan vektor dengan memotongnya menjadi potongan pendek dan mengganti tiap potongan dengan nomor entri terdekatnya dalam palet kecil, seperti melukis dengan nomor. [14] |
| **Precision@K / Recall@K** | Precision@K adalah porsi top K yang relevan; recall@K adalah porsi semua item relevan yang masuk ke top K. [2] |
| **Profiling** (*pemrofilan*) | Mengukur di mana program menghabiskan waktu, tahap demi tahap, sehingga kita mempercepat bagian yang memang lambat. [13] |
| **Propensity** (*propensitas*) | Peluang sebuah item sejak awal ditampilkan kepada pembeli. [12] |
| **Query group** (*kelompok kueri*) | Semua baris yang milik daftar satu pembeli; peringkat hanya dibandingkan di dalam satu kelompok. [10] |
| **Recall ceiling** (*batas atas recall*) | Lihat *Candidate recall*. [8] |
| **Recency** (*kebaruan*) | Seberapa baru sesuatu terjadi; event yang lebih baru biasanya lebih penting. [4] |
| **Regularisation** (*regularisasi*) | Menambahkan hukuman untuk angka besar pada loss agar model tetap sederhana dan tidak overfit. [6] |
| **Re-ranking** (*pemeringkatan ulang*) | Langkah akhir yang menerapkan aturan tingkat halaman (batas per kategori, membuang item yang sudah dibeli) pada skor model. [11] |
| **Retrieval** (*pengambilan*) | Tahap pertama: memilih secara murah beberapa ratus kandidat dari seluruh katalog. [8] |
| **Sentinel value** (*nilai penanda*) | Angka ekstrem yang sengaja dipilih (misalnya 9999) yang berarti "ini tidak pernah terjadi". [9] |
| **Session** (*sesi*) | Satu rentetan aktivitas pembeli; di sini jeda 30 menit memulai sesi baru. [13] |
| **Shard** (*pecahan*) | Satu irisan katalog yang disimpan di mesinnya sendiri, seperti satu cabang perpustakaan; kueri bertanya ke setiap shard lalu menggabungkan jawabannya. [14] |
| **SHAP** | Metode yang memecah satu prediksi menjadi kontribusi dari tiap fitur, dan kontribusi itu bila dijumlahkan sama dengan keluaran model. [10] |
| **Smoothing** (*penghalusan*) | Lihat *Bayesian average*. [3] |
| **Softmax** | Resep yang mengubah daftar skor apa pun menjadi probabilitas yang berjumlah 1, dengan skor lebih besar mendapat porsi lebih besar. [7] |
| **Sparse matrix** (*matriks jarang*) | Tabel besar yang hampir semua selnya kosong, sehingga kita hanya menyimpan sel yang terisi. [4] |
| **Sparsity** (*kejarangan*) | Porsi tabel pembeli × item yang kosong; biasanya di atas 99%. [1] |
| **SRM (Sample ratio mismatch)** (*ketidaksesuaian rasio sampel*) | Dua kelompok uji A/B ternyata berukuran berbeda dari rencana (misalnya 50.6/49.4, bukan 50/50, pada jutaan pengguna), yang menandakan ada bug sehingga hasilnya tidak bisa dipercaya. [14] |
| **Temperature** (*temperatur*) | Kenop pada softmax: temperatur rendah membuat model lebih tajam dan yakin, temperatur tinggi membuatnya lebih datar dan hati-hati. [7] |
| **TF-IDF** | Bobot kata: tinggi untuk kata yang umum di satu deskripsi tetapi jarang di semua deskripsi, sehingga kata itu mencirikan item tersebut. [4] |
| **Thompson sampling** | Metode eksplorasi: tarik satu rasio klik acak yang masuk akal untuk tiap item dari distribusi Beta-nya, lalu tampilkan item dengan tarikan tertinggi. [11] |
| **Training/serving skew** (*ketidaksesuaian pelatihan/penyajian*) | Ketidakcocokan antara cara fitur dihitung untuk pelatihan dan cara fitur dihitung saat langsung, yang diam-diam merugikan model. [14] |
| **Two-tower model** (*model dua menara*) | Model retrieval dengan satu jaringan saraf kecil untuk pembeli dan satu untuk item, yang vektor keluarannya dibandingkan dengan dot product. [7] |
| **TTL (Time to live)** (*masa hidup*) | Berapa lama jawaban tersimpan (cache) boleh dipakai ulang sebelum harus dihitung ulang. [14] |
| **Vector** (*vektor*) | Daftar angka dalam urutan tetap; kita mengubah tiap produk atau pembeli menjadi satu agar komputer bisa membandingkannya. [4] |
| **Weak learner** (*pembelajar lemah*) | Model yang sangat sederhana, seperti pohon dengan satu pertanyaan dan dua jawaban, yang hanya sedikit lebih baik daripada menebak dan berguna hanya bila jumlahnya banyak. [10] |
