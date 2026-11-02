# Lampiran A — Kunci jawaban kuis

Jawaban untuk setiap kuis di akhir bab, lengkap dengan penjelasan singkat. Latihan praktik menjelaskan apa yang seharusnya Anda amati; angka persis Anda bergantung pada perubahan yang Anda buat.

## Bab 1 — Apa yang Dilakukan Sistem Rekomendasi

**Q1.1** — B. Retrieval dengan murah menyaring jutaan item menjadi beberapa ratus kandidat dan dinilai berdasarkan recall. Opsi A menggambarkan ranking, C menggambarkan re-ranking, dan D tidak berkaitan dengan serving. Apa pun yang terlewat oleh retrieval tidak akan pernah bisa dipulihkan oleh tahap-tahap setelahnya.

**Q1.2** — C. Dwell time dan add-to-cart adalah produk sampingan dari perilaku normal, bukan pernyataan pendapat yang disengaja. Rating bintang, ulasan tertulis, dan ketukan jempol ke atas semuanya adalah umpan balik eksplisit karena pengguna memang memilih untuk menyatakan preferensi.

**Q1.3** — B. Platform sungguhan hanya melihat log, sehingga simulator hanya menulis apa yang akan dicatat oleh sebuah aplikasi dan merahasiakan variabel laten. Karena itu, metode harus memulihkan sinyal dari jenis data yang sama dengan yang akan mereka lihat di produksi. Karena kebenaran dasarnya ada, kita juga tahu bahwa datanya memang dapat dipelajari, yang membantu saat mendiagnosis metode yang gagal.

**Q1.4** — CTR adalah klik dibagi impresi, dan jumlah impresi adalah penyebut yang menunjukkan seberapa sering sebuah item benar-benar ditampilkan. Jika hanya klik yang dicatat, item yang tampak populer mungkin hanya sangat sering ditampilkan, dan Anda tidak bisa membedakan "ditampilkan lalu diabaikan" dari "tidak pernah ditampilkan". Anda juga tidak akan punya negatif untuk pelatihan dan tidak bisa mempelajari position bias.

**Q1.5** — Sparsity adalah fraksi sel pengguna-item yang tidak memiliki interaksi tercatat (di sini 99.388%). Setiap pengguna hanya menyentuh sangat sedikit item, sehingga sebuah algoritma harus menyimpulkan selera dari segelintir sel yang terisi. Akibatnya, metode berbagi kekuatan statistik antar pengguna dan item: metode neighbourhood (bab 5), faktorisasi low-rank (bab 6), dan fitur konten. Jawaban apa pun yang menyebut salah satu konsekuensi tersebut benar.

**Q1.6** — Hari sale pada hari Sabtu memiliki bobot mentah 1.2 × 3.0 = 3.6. Hari gajian pada hari Minggu yang bukan hari sale memiliki 1.2 × 1.4 = 1.68. Flag dikembalikan karena loop simulasi memakainya kembali: pada hari sale, diskon efektif dinaikkan 0.2, dan probabilitas order memiliki bonus gajian dan sale, sehingga waktu mengubah perilaku, bukan hanya volume trafik.

**Q1.7** — 251 order. Kalikan ketiga rasio: 0.1059 × 0.4295 × 0.5517 ≈ 0.02509 (2.51% impresi yang berakhir menjadi order), dan 10,000 × 0.02509 ≈ 250.9, yang dibulatkan menjadi 251.

**Q1.8** — Kepadatan = 50,000 / (1,000 × 10,000) = 50,000 / 10,000,000 = 0.5%, sehingga sparsity = 99.5%. Bahkan contoh yang "tampak padat" ini membiarkan 199 dari setiap 200 sel kosong.

**Q1.9** — Setelah perubahan, skrip mencetak bahwa 50% teratas (1,250 item) menerima 92.8% interaksi, sehingga separuh bawah katalog hanya berbagi sekitar 7% darinya. Top 5% (125 item) mendapat 35.1%, yang terletak di antara 12.5% untuk top 1% dan 50.3% untuk top 10%. Ini menunjukkan betapa curam interaksi terkonsentrasi di bagian kepala dan betapa tipisnya bagian ekor, itulah sebabnya baseline popularitas kuat.

**Q1.10** — Dua alasan yang baik: (1) pengguna mungkin tidak pernah menyadari item tersebut karena berada di bagian bawah daftar atau di luar layar (exposure dan position bias), sehingga tidak ada klik tidak berarti tidak ada minat; (2) pengguna mungkin tertarik tetapi memilih item lain lebih dulu atau berniat kembali nanti. Memperlakukan ini sebagai negatif keras mengajari model untuk mengulang apa yang ditampilkan sebelumnya dan menghukum item karena penempatannya yang buruk.

## Bab 2 — Mengukur keberhasilan

**Q2.1** — B. Dengan pengacakan acak, sebagian event test lebih tua daripada event pelatihan, sehingga model belajar dari masa depan (lonjakan popularitas, selera pengguna di kemudian hari, event lain dari sesi yang sama). A bukan masalahnya, C tidak masuk akal, dan D tidak berkaitan dengan cara data dipisahkan. Pemisahan berbasis waktu menghilangkan kebocoran ini.

**Q2.2** — C. Kedua daftar berisi himpunan item yang sama, sehingga recall@10, precision@10, dan coverage identik. NDCG mendiskon berdasarkan posisi, sehingga daftar yang menempatkan item yang dipesan (gain 7) di urutan pertama mendapat skor lebih tinggi daripada yang menempatkannya di urutan terakhir.

**Q2.3** — B. Candidate recall adalah porsi positif masa depan yang muncul di himpunan kandidat, sehingga tidak ada ranker yang hanya mengurutkan ulang kandidat tersebut yang bisa melampauinya. Ia tidak "selalu lebih rendah" daripada recall akhir (A salah: ia adalah batas atas), ia bukan ukuran keluasan (C adalah coverage), dan meningkatkannya membutuhkan retriever yang lebih baik, bukan ranker yang lebih baik (D).

**Q2.4** — Karena AP juga harus menghukum positif yang tidak pernah muncul. Positif yang terlewat tidak menambah apa pun ke pembilang tetapi tetap dihitung di penyebut $R$. Membagi dengan jumlah hit akan memberi hadiah yang sama pada daftar yang menemukan satu item di peringkat 1 dan mengabaikan empat lainnya seperti pada daftar yang menemukan kelimanya.

**Q2.5** — Dengan inner join (atau merge yang dimulai dari `recs`), positif masa depan yang tidak pernah direkomendasikan akan hilang dari tabel. Jumlah baris per pengguna akan menjadi jumlah hit, bukan $R$, sehingga recall menjadi hits/hits = 1 untuk setiap orang yang punya hit, dan pengguna tanpa hit akan lenyap alih-alih mendapat skor 0. Left merge dari label menjaga setiap positif tetap ada di penyebut.

**Q2.6** — `dropna(subset=["pos"])` membuang positif masa depan yang tidak direkomendasikan di top $k$ (`pos`-nya NaN setelah left merge), sehingga hanya hit yang berkontribusi pada DCG. Konstanta `+ 2` dipakai karena `pos` berbasis nol sedangkan rumus memakai peringkat $i$ berbasis satu: $\log_2(i+1) = \log_2(\text{pos}+2)$. Memakai `+ 1` akan membagi dengan $\log_2 1 = 0$ pada slot pertama.

**Q2.7** — DCG@3 = 4.131, IDCG@3 = 7.631, NDCG@3 = 0.541. DCG = $0/\log_2 2 + 1/\log_2 3 + 7/\log_2 4 = 0.631 + 3.5 = 4.131$. Urutan ideal menempatkan order di urutan pertama: $7/\log_2 2 + 1/\log_2 3 = 7 + 0.631 = 7.631$. Rasionya $4.131/7.631 \approx 0.541$.

**Q2.8** — precision@5 = 0.4 dan recall@5 = 0.5. Dua hit dalam lima slot menghasilkan $2/5 = 0.4$; dua dari empat positif masa depan ditemukan, menghasilkan $2/4 = 0.5$.

**Q2.9** — NDCG adalah rata-rata kualitas ranking atas semua pengguna dan tidak melihat keluasan atau batas atas pipeline. Coverage menyingkap model yang menampilkan segelintir bestseller yang sama kepada semua orang (NDCG baik pada pekan yang didominasi popularitas, tetapi hanya sebagian kecil katalog yang pernah ditampilkan). Candidate recall menyingkap bahwa tahap pertama tidak pernah mengambil banyak item yang kemudian dibeli pengguna, sehingga ranker tidak punya apa-apa untuk dipilih dan peningkatan NDCG dari penyetelan ulang akan terbatas; NDCG saja tidak memberi tahu Anda apakah hambatannya ada di retrieval atau ranking.

**Q2.10** — Coverage@20 seharusnya turun menjadi sekitar $20/2311 \approx 0.0087$, karena hanya 20 item berbeda yang pernah ditampilkan ke semua pengguna (jumlah item yang tersedia adalah 2311 pada output yang tercetak). Angka akurasi akan berbeda dari run acak, tetapi tidak akan dipersonalisasi. Latihan ini menunjukkan bahwa coverage dan akurasi adalah sumbu yang terpisah: acak punya coverage 1.0 dengan akurasi hampir nol, dan daftar konstan punya coverage hampir nol.

## Bab 3 — Baseline popularitas

**Q3.1** — B. Dalam simulator, setiap item memiliki kualitas tersembunyi yang tetap, sehingga popularitas tidak bergeser. Ketika popularitas yang mendasari stabil, hitungan 53 hari adalah estimasi yang lebih sedikit noise-nya daripada hitungan 7 hari. Di marketplace nyata dengan flash sale dan musiman, jendela pendek sering menang, jadi setel pada data held-out.

**Q3.2** — C. Dengan $v=0$, bobot pada rating item itu sendiri, $v/(v+m)$, adalah nol, sehingga skor sama dengan rata-rata global $C$ (4.324 pada cetakan). Nilai 4.4 atau 4.5 yang disebutkan hanyalah placeholder yang ditulis untuk item baru. Smoothing memperlakukannya sebagai "tidak ada informasi".

**Q3.3** — B. Coverage@20 adalah jumlah item berbeda yang direkomendasikan dibagi katalog yang tersedia. Peringkat global memberikan 20 item yang sama kepada semua orang, sehingga hanya 20 item yang pernah ditampilkan. Inilah akar popularity bias dan feedback loop.

**Q3.4** — Berikan `num` = total bintang (R · v), `den` = jumlah rating v, `prior` = rata-rata global C, dan `strength` = m. Fungsi kemudian mengembalikan (R·v + C·m)/(v + m). Memecah pecahan tersebut menghasilkan v/(v+m)·R + m/(v+m)·C, yaitu rata-rata Bayesian. Panggilan yang sama berlaku untuk CTR (klik, impresi, CTR global, pseudo-impresi).

**Q3.5** — Item yang ditampilkan di atas mendapat lebih banyak impresi dan klik, yang menaikkan skor popularitasnya, yang menjaganya tetap di atas, sementara item yang tidak pernah ditampilkan tidak bisa mengumpulkan bukti. Hal-hal berikut melemahkannya: personalisasi, eksplorasi (ε-greedy, Thompson sampling), rasio yang dinormalisasi terhadap exposure seperti CTR yang di-smooth, koreksi position bias, atau re-ranking untuk diversitas.

**Q3.6** — Tidak. Filternya adalah `age_days < window_days`, ketat, sehingga event dengan usia tepat 7.0 dikecualikan. Setiap interaksi yang lolos mempertahankan bobot penuh `w` (1, 3, atau 5) tanpa peluruhan, karena ini adalah jendela keras. Mengganti `<` menjadi `<=` akan menyertakan event berusia tepat 7 hari; untuk menyertakan seluruh hari ke-8 Anda akan memanggilnya dengan `window_days=8` (atau memakai `<= 8`).

**Q3.7** — Sekitar 4.98. Order: 5 · 2^(−7/7) = 5 · 0.5 = 2.5. Klik: 1 · 2^(−14/7) = 0.25. Add-to-cart: 3 · 2^(−3/7) ≈ 3 · 0.743 = 2.229. Jumlah = 2.5 + 0.25 + 2.229 ≈ 4.98.

**Q3.8** — Dengan v = 20: 20/120·4.8 + 100/120·4.0 = 0.8 + 3.333 = 4.133. Dengan v = 400: 400/500·4.8 + 100/500·4.0 = 3.84 + 0.80 = 4.64. Semakin banyak rating menggeser estimasi dari prior (4.0) menuju rata-rata item itu sendiri (4.8).

**Q3.9** — Half-life = 10 hari / 2 = 5 hari, karena 25% = 2^(−2), yaitu dua half-life telah berlalu pada usia 10. Half-life yang lebih kecil memusatkan bobot pada event terbaru, sehingga skor bereaksi cepat terhadap perilaku baru tetapi bertumpu pada lebih sedikit event efektif, yang membuatnya lebih berisik.

**Q3.10** — Dengan half-life yang sangat pendek (2 hari), pop_decay berperilaku seperti jendela pendek: perilakunya seharusnya bergerak menuju baris pop_7d yang berumur pendek, karena bertumpu pada sedikit event efektif. Dengan half-life yang sangat panjang (30 hari), bobot mendekati 1 untuk seluruh riwayat 53 hari, sehingga pop_decay mendekati pop_all_time. Ini sesuai dengan penjelasan bahwa, dengan popularitas simulasi yang stabil, lebih banyak riwayat efektif berarti lebih sedikit noise. Nilai persisnya bergantung pada run Anda; urutannya yang perlu diperiksa.

## Bab 4 — Content-based filtering

**Q4.1** — B. IDF besar untuk term yang langka dan kecil untuk term yang muncul di banyak dokumen, sehingga kata seperti "sneakers" bernilai lebih daripada "home". Preferensi panjang (A) bukan tujuan; normalisasi justru menghilangkannya, dan TF-IDF tidak membuat vektor menjadi ortogonal atau menghapus stop-word dengan sendirinya.

**Q4.2** — B. Untuk vektor satuan, penyebut cosine adalah 1, sehingga cosine sama dengan dot product, itulah sebabnya satu perkalian sparse `X @ X.T` menghasilkan semua kemiripan berpasangan. Ia bukan jarak Euclidean (meskipun keduanya monoton terkait untuk vektor satuan), ia non-negatif untuk vektor TF-IDF, dan normalisasi menghilangkan ketergantungan pada panjang teks.

**Q4.3** — C. Scorer content-based hanya membutuhkan atribut item itu sendiri, yang ada sejak saat item didaftarkan. Popularitas, co-visitation, dan metode berbasis order semuanya membutuhkan interaksi yang belum dimiliki item baru.

**Q4.4** — Kedua item memiliki teks yang identik setelah tokenisasi (kata judul, kategori, dan merek yang sama), sehingga vektor TF-IDF mereka identik dan sudut di antaranya nol. Rekomender yang murni berbasis konten tidak bisa membedakan keduanya dan mungkin mengisi slot teratas dengan salinan produk yang sama; memecah seri membutuhkan sinyal perilaku seperti harga, rating, popularitas, atau kualitas penjual, atau langkah diversitas.

**Q4.5** — Over-specialisation berarti sistem hanya merekomendasikan item yang mirip dengan riwayat pengguna, sehingga tidak pernah mengejutkan mereka atau menjangkau kebutuhan baru. Solusinya: padukan dengan sinyal kolaboratif seperti co-visitation, lakukan re-rank untuk diversitas (MMR atau batas per kategori), atau sisihkan slot untuk eksplorasi. Dua di antaranya sudah cukup.

**Q4.6** — Tanpa guard, pengguna tanpa event positif memiliki `wsum == 0` dan `1 / wsum` menjadi tak hingga, sehingga mengalikan baris yang seluruhnya nol dengannya menghasilkan NaN. Dengan guard, pembaginya 1, baris `W @ X` tetap nol semua, dan `normalize` meninggalkan vektor nol. Profil menjadi nol semua dan setiap item mendapat skor 0, itulah sebabnya skrip menambahkan tie-break popularitas kecil untuk pengguna-pengguna ini.

**Q4.7** — 0.571. Item A memiliki bobot 1 × 2^0 = 1. Item B memiliki bobot 3 × 2^(−14/7) = 3 × 0.25 = 0.75. Porsi A adalah 1 / (1 + 0.75) = 0.5714, yaitu 0.571. Event cart yang lebih lama, meskipun kekuatannya lebih tinggi, berkontribusi lebih sedikit daripada klik yang baru.

**Q4.8** — 0.48. Vektornya berpanjang satu, sehingga cosine adalah dot product: 0.6 × 0.8 + 0.8 × 0 + 0 × 0.6 = 0.48. Keduanya hanya berbagi dimensi pertama, itulah sebabnya kemiripannya sedang, bukan tinggi.

**Q4.9** — Pengamatan yang diharapkan: half-life yang sangat pendek seperti 1 hari membuang sebagian besar riwayat, sehingga profil bertumpu pada sedikit event dan kualitas biasanya turun; yang sangat panjang seperti 30 hari mendekati rata-rata tanpa bobot dan mungkin lambat bereaksi, sehingga half-life moderat di dekat baseline 7 hari biasanya terbaik. Angka aktual Anda mungkin berbeda, karena minat dalam simulator cukup stabil. Apa pun yang Anda amati, bandingkan ndcg@20 dan recall@20 dengan 0.0909 dan 0.1378 dan perlakukan perbedaan dalam beberapa perseribu dengan hati-hati.

## Bab 5 — Collaborative filtering

**Q5.1** — B. CF berbasis pengguna mengasumsikan bahwa orang yang sepakat di masa lalu kemungkinan akan sepakat lagi, sehingga item yang disukai pengguna dengan riwayat yang beririsan adalah rekomendasi yang baik. Ia tidak melihat judul (A), ia tidak hanya soal popularitas (C), dan ia tidak mengasumsikan selera statis (D) — itulah sebabnya peluruhan waktu dipakai di bagian lain buku ini.

**Q5.2** — C. CF item-ke-item menghitung tabel item-serupa item–item yang mahal secara offline. Saat request, ia hanya mencari tetangga dari item di riwayat pengguna, sehingga biayanya tidak bergantung pada ukuran katalog dan jumlah pelanggan. User-kNN harus membandingkan target dengan semua pengguna pada setiap request.

**Q5.3** — B. Jaccard hanya bertanya "ada atau tidak", yang cocok untuk sinyal biner seperti keanggotaan sesi yang sama dan mencegah pengguna berat mendominasi lewat hitungan. Ketika bobot membawa informasi (order = 5 klik), cosine pada matriks berbobot mempertahankan informasi itu. Kedua ukuran bersifat simetris.

**Q5.4** — Setiap baris dinormalisasi sehingga bobot keluar dari setiap item sumber berjumlah 1. Tanpa itu, pasangan yang melibatkan item yang sangat populer akan memiliki hitungan mentah yang besar dan mendominasi setiap daftar kandidat terlepas dari seberapa kuat item-item itu sebenarnya berhubungan. Normalisasi membuat bobot sebanding antar item sumber dan mengurutkan tetangga setiap item berdasarkan kekuatan relatif; ia juga membuat matriks menjadi asimetris.

**Q5.5** — Kedua item memiliki sedikit interaksi dan beberapa pengguna bersama, sehingga cosine menggelembung akibat dukungan yang sangat kecil (kemiripan maksimum tabel adalah 1.000, biasanya satu pengguna bersama). Perbaikan: wajibkan jumlah minimum pengguna bersama (support threshold), kecilkan kemiripan dengan n/(n+λ), atau batasi/sampel datanya. Salah satunya sudah cukup.

**Q5.6** — Ia menggabungkan semua event dari item yang sama dalam satu sesi menjadi satu bobot terkuat, sehingga me-refresh halaman sepuluh kali dihitung sekali. Dengan `.sum()`, item yang diklik, dimasukkan keranjang, dan dipesan dalam satu sesi akan dihitung 1+3+5 = 9, bukan 5, dan loop klik berulang atau bot akan menggelembungkan bobot pasangan, mengacaukan skala kekuatan yang dimaksudkan.

**Q5.7** — Cosine 0.632, Jaccard 0.333. Dot product-nya 2·1 + 0·1 + 1·0 = 2; norm-nya √5 = 2.236 dan √2 = 1.414, sehingga cosine = 2/3.162 = 0.632. Himpunan penggunanya X = {u1, u3} dan Y = {u1, u2}: irisan 1, gabungan 3, sehingga Jaccard = 1/3 = 0.333.

**Q5.8** — 0.55. Skor(C) = Σ recency(seed) · w(seed→C) = 0.5·0.3 + 1.0·0.4 = 0.15 + 0.40 = 0.55. Ini adalah penjumlahan dalam `covisit` (`cand_recent_score * w`, dikelompokkan dan dijumlahkan per pengguna dan kandidat). Dua seed yang menunjuk ke kandidat yang sama saling memperkuat.

**Q5.9** — B. item_knn memiliki coverage@20 0.8399 (tertinggi), dan ndcg@20 0.0925, di bawah user_knn (0.1662) dan covisit (0.1197). user_knn tidak memiliki coverage tertinggi (A salah); recall@5 covisit (0.0795) di bawah milik user_knn (0.1230) (C salah); ndcg pop_7d (0.0475) di bawah milik item_knn (D salah).

**Q5.10** — Dalam run percobaan, ndcg@20 berubah 0.1662 (k=50), 0.1702 (k=100), 0.1706 (k=200), 0.1700 (k=400) dan recall@20 0.2273, 0.2419, 0.2409, 0.2405: peningkatan berhenti setelah sekitar k=100 dan kurvanya datar atau sedikit turun. Setelah titik tertentu, tetangga tambahan hanyalah pengguna yang mirip secara lemah yang suaranya menambah noise alih-alih sinyal, dan bobot kemiripan membuat kontribusi mereka kecil.

## Bab 6 — Faktorisasi matriks (*matrix factorisation*)

**Q6.1** — B. Faktor laten adalah satu dimensi yang dipelajari dari vektor pengguna dan item. Tidak ada yang melabelinya sebelumnya; optimizer menemukannya karena memberi setiap pengguna dan item sebuah nilai pada dimensi itu membantu merekonstruksi interaksi. Ia bukan kolom matriks mentah, hyperparameter, atau atribut yang dimasukkan penjual.

**Q6.2** — B. $p_{ui}$ adalah target biner "apakah pengguna ini menunjukkan minat" ($1[R_{ui}>0]$), dan $c_{ui}=1+\alpha R_{ui}$ membobot seberapa penting galat kuadrat pada sel itu. Sel yang tidak teramati mempertahankan $c=1$, sehingga mereka adalah nol yang lemah, sedangkan order dan cart adalah positif yang kuat.

**Q6.3** — C. $Y^\top Y$ tidak bergantung pada pengguna, sehingga dihitung sekali per sweep. Koreksi $Y^\top(C^u-I)Y$ hanya memiliki baris non-nol untuk $n_u$ item yang disentuh pengguna, sehingga biaya per pengguna berskala dengan riwayat mereka, bukan katalog. Loss-nya tetap tidak konveks secara bersama-sama, dan trik ini tidak mengatakan apa-apa tentang regularisasi atau rating eksplisit.

**Q6.4** — Sel kosong sebagian besar berarti "belum dilihat", bukan "tidak disukai". Mencocokkannya sebagai nol sungguhan akan mengklaim bahwa hampir seluruh katalog dibenci oleh setiap pengguna dan akan mendorong model memprediksi hampir nol di mana-mana. ALS implisit mempertahankannya sebagai nol tetapi hanya memberinya confidence dasar $c=1$, sementara sel yang teramati mendapat $c=1+\alpha R$, sehingga loss terutama peduli untuk menjelaskan interaksi yang kita yakini.

**Q6.5** — Dengan vektor mendekati nol, setiap selisih $x_u\cdot(y_i-y_j)$ sekitar 0. Maka $\sigma(0)=0.5$ dan loss $-\ln 0.5=\ln 2\approx 0.693$ untuk setiap triple. Melihat nilai ini pada epoch 1 adalah pemeriksaan kewarasan bahwa model dimulai sebagai lemparan koin tanpa informasi dan loss tersambung dengan benar.

**Q6.6** — $x_u = 23/34 \approx 0.676$. Confidence-nya $c=1+4R=(5,13)$. Pembilang: $\sum c\,p\,y = 5\cdot1\cdot2+13\cdot1\cdot1=23$. Penyebut: $\sum c\,y^2+\lambda = 5\cdot4+13\cdot1+1=34$. Pembagiannya menghasilkan 0.676.

**Q6.7** — $\sigma(-\ln 3)=1/(1+3)=0.25$; loss $=-\ln0.25=\ln4\approx1.386$; $s=\sigma(\ln 3)=0.75$. Selisihnya negatif, sehingga pasangan tersebut berurutan salah (negatif mengalahkan positif). Pengali $s=0.75$ besar, sehingga pembaruannya besar, yang persis merupakan perilaku yang diinginkan BPR untuk pasangan yang urutannya salah.

**Q6.8** — `sig` sama dengan $\sigma(-\text{gap}) = 1-\sigma(\text{gap})$, yaitu $-\partial\,\text{loss}/\partial\,\text{gap}$. Dengan aturan rantai, setiap gradien adalah `sig` dikali turunan gap terhadap parameter tersebut: $y_i-y_j$ untuk pengguna, $x_u$ untuk item positif, dan $-x_u$ untuk item negatif. `gj` negatif karena menaikkan gap mengharuskan menurunkan skor item negatif, sehingga $y_j$ bergerak menjauh dari vektor pengguna. Pasangan yang sudah berurutan benar memiliki `sig` mendekati 0 dan hampir tidak mengubah apa pun.

**Q6.9** — Recall@20 hanya bertanya apakah item masa depan pengguna muncul di mana saja dalam top 20; NDCG juga memberi hadiah untuk menempatkannya di dekat puncak. ALS menemukan lebih banyak item yang benar, tetapi mengurutkannya kurang tajam daripada user-kNN. Profil itu cocok untuk generator kandidat (dinilai dengan recall) yang memasok tahap ranking berikutnya (dinilai dengan NDCG), seperti ranker LightGBM di repo ini.

**Q6.10** — Angka persisnya bergantung pada run Anda (skrip di-seed dan deterministik), jadi bandingkan dengan ndcg@20 0.1581 untuk `als_implicit` di `results/ch06_mf.txt`. Harapkan `alpha=1.0` mengubah ndcg@20 secara nyata, karena dengan alpha kecil sel teramati dan tak teramati membawa confidence yang hampir sama dan model berperilaku berbeda; arahnya tidak dijamin, jadi periksalah. Harapkan `factors=8` turun jauh di bawah run 32-faktor: 8 dimensi tidak bisa memisahkan selera yang tersebar di banyak kategori dan ribuan item. Loss ALS akhir tidak sebanding antar nilai `alpha` karena loss itu sendiri berubah bersama bobotnya. Pelajarannya adalah bahwa menyapu hyperparameter terhadap label evaluasi berarti menyetel pada test set; gunakan jendela validasi yang lebih awal dan laporkan pekan test sekali saja.

## Bab 7 — Embedding dan retrieval two-tower

**Q7.1** — B. Membagi cosine dengan $\tau$ kecil meregangkan selisih antar skor, sehingga softmax menjadi lebih tajam dan loss terkonsentrasi di puncak ranking, menghukum negatif yang sulit. Softmax yang lebih datar (A) adalah yang dihasilkan temperatur tinggi. Temperatur tidak mengubah dimensi embedding atau isi batch.

**Q7.2** — C. Sebuah batch adalah sampel acak pasangan positif, sehingga setiap item muncul sebanding dengan jumlah interaksinya. Item populer karenanya muncul sebagai negatif di banyak baris dan menerima dorongan ke bawah yang lebih besar, meskipun serving menilai semua item secara seragam. Normalisasi L2 menetapkan norm menjadi 1, sehingga bukan penyebabnya.

**Q7.3** — A. Vektor item hanya bergantung pada atribut item, sehingga dapat dihitung sekali dan disimpan dalam indeks. Skornya adalah inner product, yang memang untuk itulah struktur ANN dibangun. Tower yang menerima riwayat pengguna sebagai input item (D) akan merusak ini.

**Q7.4** — Ia mengurangi $\log q_j$, log dari probabilitas item $j$ terambil ke dalam batch, dari logit kolom $j$. Item populer ($q_j$ besar) mendapat dorongan kecil dan item langka dorongan besar, yang membatalkan dorongan ke bawah ekstra yang diterima item populer karena di-oversample sebagai negatif. Koreksi ini hanya dipakai saat pelatihan; saat serving dipakai dot product biasa.

**Q7.5** — Ia mengukur berapa banyak dari top-20 brute-force yang tepat masih dikembalikan oleh pencarian aproksimasi. Indeks ANN hanya perlu mengaproksimasi pencarian eksak, sehingga kesetiaan terhadap brute force memisahkan galat indeks dari kualitas model. NDCG akan mencampur keduanya dan tidak bisa memberi tahu mana yang kehilangan akurasi.

**Q7.6** — Ia mengganti kemunculan item target dalam riwayat pengguna itu sendiri dengan indeks padding, sehingga user tower tidak bisa melihat jawabannya. Jika dihapus, model bisa menyalin target dari riwayat, sehingga training loss akan turun lebih cepat dan tampak lebih baik, tetapi evaluasi pada pekan held-out, di mana jalan pintas seperti itu tidak ada, akan lebih buruk. Ini adalah contoh kecil target leakage.

**Q7.7** — Sekitar 43,159 dot product, atau sekitar 0.43% dari brute force. Penilaian centroid berbiaya 4,096; setiap cluster memiliki $10{,}000{,}000 / 4{,}096 \approx 2{,}441$ item, sehingga 16 probe memindai $16 \times 2{,}441 = 39{,}062.5$. Totalnya $4{,}096 + 39{,}062.5 = 43{,}158.5$, dan $43{,}158.5 / 10{,}000{,}000 \approx 0.43\%$, kira-kira 230 kali lebih sedikit pekerjaan.

**Q7.8** — Sebesar $\ln(0.04 / 0.0004) = \ln 100 \approx 4.605$ satuan logit. Koreksi menambahkan $-\ln q$: $-\ln 0.04 = 3.219$ untuk item populer dan $-\ln 0.0004 = 7.824$ untuk item langka, dan selisihnya 4.605. Dalam bentuk probabilitas, skor ter-eksponensial item yang lebih langka dikalikan 100 relatif terhadap item populer.

**Q7.9** — Pengamatan yang diharapkan (verifikasi pada run Anda): spesifikasi `two_tower` tetap jauh di bawah `two_tower_logq` pada ndcg@20 di setiap temperatur, karena popularity bias berasal dari skema sampling, bukan temperatur. Temperatur mengubah ketajaman loss, sehingga ndcg dan terutama coverage@20 akan bergeser antar run; nilai yang sangat rendah seperti 0.02 cenderung lebih sulit dioptimalkan. Inti latihan ini adalah bahwa selisih logQ bertahan dan temperatur hanyalah tombol penyetelan, bukan obatnya. Pulihkan hasil dengan `git checkout book/results`.

## Bab 8 — Pembangkitan kandidat multi-sumber

**Q8.1** — B. Item yang tidak ada di himpunan kandidat tidak akan pernah mencapai ranker, sehingga tahap pertama dinilai dari berapa banyak positif masa depan yang dikandungnya. False positive murah karena tahap kedua, dengan fitur yang lebih kaya, menyingkirkannya. Ranker tidak membutuhkan input yang bersih (D salah), dan precision dapat dihitung pada umpan balik implisit (A salah).

**Q8.2** — C, `catpop`. Unique pair recall-nya 0.0968, dibandingkan 0.0524 untuk `covisit`, 0.0298 untuk `recent`, dan 0.0253 untuk `global_trending`. Ia juga memiliki recall mandiri tertinggi (0.3274), karena menjangkau pengguna lewat klik dan pencarian.

**Q8.3** — B. Positive rate naik dari 0.0090 menjadi 0.1817 seiring semakin banyak sumber yang sepakat, sehingga jumlah sumber dan skor setiap sumber adalah fitur yang informatif. Membuang kandidat bersumber tunggal (A, C) akan membuang sebagian besar recall, karena 76,612 kandidat hanya memiliki satu sumber. Menghitung sumber tidak bisa menggantikan ranker (D), yang juga memakai fitur pengguna, item, dan cross.

**Q8.4** — C. Hanya `global_trending`. `recent` kosong tanpa riwayat, `covisit` di-seed dari `recent` sehingga juga kosong, dan `catpop` membutuhkan category affinity dari klik atau pencarian. Outer join hanya mengisi kolom skor yang hilang; ia tidak mengarang kandidat (A, D).

**Q8.5** — Gabungan berisi rata-rata 103.94 kandidat per pengguna, sehingga memotong daftar berperingkat pada K=100 membuang kandidat di ujung daftar yang lebih panjang, dan sebagian di antaranya adalah positif. Batas atas 0.4645 adalah recall dari seluruh himpunan kandidat tanpa pemotongan. Recall@K hanya bisa mencapainya ketika K setidaknya sebesar ukuran pool untuk setiap pengguna.

**Q8.6** — Menyembunyikan adalah umpan balik negatif eksplisit, dan fitur yang dipelajari hanya mengubah probabilitas: model masih bisa sesekali menampilkan item yang disembunyikan. Filter keras pada pool menjamin item itu tidak pernah muncul lagi, lebih murah (lebih sedikit baris untuk dinilai) dan mudah diaudit. Pengguna menoleransi rekomendasi yang tidak relevan tetapi tidak yang telah mereka tolak secara eksplisit.

**Q8.7** — Nilainya akan menjadi 3, bukan 1. Setelah penukaran, ketiga kolom skor sudah terisi 0.0, sehingga `notna().sum(axis=1)` menghitung ketiganya untuk setiap baris, dan suku trending tidak menambah apa-apa karena item tidak ada dalam daftar itu. Fitur tersebut tidak lagi memisahkan kandidat bersumber tunggal dari multi-sumber dan akan kehilangan sinyal yang menaikkan positive rate dari 0.0090 menjadi 0.1817.

**Q8.8** — Item yang kehabisan stok di top 30 akan dihapus setelah `nlargest`, menyisakan kurang dari 30 item trending, dan daftar akan menyusut setiap kali item populer habis terjual. Memfilter dulu lalu mengambil top 30 selalu mengembalikan 30 item yang tersedia (jika jumlahnya cukup).

**Q8.9** — 12.1880 + 35.8321 + 54.9102 + 30.0000 = 132.9303, dikurangi ukuran gabungan 103.9366 menghasilkan 28.9937, yaitu sekitar 29.0 proposal (pengguna, item) duplikat per pengguna, yang merupakan 28.9937 / 132.9303 = 21.8% dari jumlah. Itulah tumpang tindih antar sumber, yang dihitung oleh `cand_n_sources`.

**Q8.10** — Harapkan `cands_per_user` `catpop` naik (lebih banyak kategori digabungkan, hingga 5 × 20 item per pengguna dibanding 3 × 20) dan recall-nya naik sedikit, karena positif masa depan pengguna kadang berada di kategori ke-4 dan ke-5. Kandidat per pengguna dan recall gabungan naik sedikit, dengan hasil yang semakin berkurang karena kategori tambahan memiliki porsi affinity yang rendah. Ranker kemudian harus menilai lebih banyak baris per request, yang menaikkan latensi, dan sebagian besar kandidat tambahan adalah negatif, sehingga tugas ranker memisahkan positif dari negatif menjadi lebih sulit. Hasilnya ditimpa oleh run tersebut, karena itu `git checkout` untuk memulihkannya.

## Bab 9 — Rekayasa fitur untuk pemeringkatan

**Q9.1** — B. Point-in-time correctness berarti setiap fitur untuk sebuah contoh pelatihan dihitung hanya dari informasi yang ada sebelum waktu cutoff contoh tersebut. Normalisasi, jumlah event yang sama, dan frekuensi refresh adalah pilihan rekayasa yang tidak berkaitan. Melanggar aturan ini disebut leakage dan menggelembungkan metrik offline.

**Q9.2** — B. Event yang ber-timestamp tepat pada cutoff termasuk jendela label `[cutoff, cutoff + days)`, yang dipilih `data.labels` dengan `ts >= cutoff`. Operator `<` yang ketat dalam `History.at` karenanya menempatkan setiap event tepat di satu sisi batas, sehingga tidak ada event yang sekaligus menjadi input fitur dan label.

**Q9.3** — B. Dengan smoothing, satu impresi dan satu klik dicampur dengan `s` impresi imajiner pada rate global, sehingga estimasi mendarat dekat `p0`. Ia bergerak menjauh hanya ketika impresi nyata terkumpul dan mengalahkan prior. Rate tetap bergantung pada klik item itu sendiri; ia hanya mempercayainya sebanding dengan buktinya.

**Q9.4** — Nol akan berarti "baru saja disentuh", yang merupakan sinyal recency terkuat, sehingga akan mengacaukan "tidak pernah" dengan "tepat menit ini". Sentinel besar seperti 9999 terletak jauh di luar rentang nyata, sehingga sebuah tree dapat memisahkan "tidak pernah disentuh" dari setiap nilai nyata dengan satu threshold. (Model linear dan neural membutuhkan flag missing yang eksplisit.)

**Q9.5** — Fitur item mendeskripsikan item untuk setiap pengguna (mis. `item_ctr_30d`). Fitur pengguna mendeskripsikan pengguna terlepas dari item (mis. `user_clicks_30d`). Fitur cross adalah agregat atas sepasang kunci (mis. `ui_clicks`, `uc_click_share_30d`, `us_orders`). Fitur cross membawa sebagian besar personalisasi, karena hanya merekalah yang berbeda antara dua pengguna untuk item yang sama. Fitur item sebagian besar membawa sinyal kualitas.

**Q9.6** — Ya, tes akan gagal. Tes menulis ulang setiap event pasca-cutoff menjadi order item 0 dalam tabel yang dimanipulasi. Fitur yang membaca seluruh tabel tanpa filter karenanya akan berubah nilainya (item 0 akan mendapat banyak order, dan item lain akan kehilangan miliknya), sehingga frame yang dibangun ulang akan berbeda dari aslinya. `pd.testing.assert_frame_equal(before, _build(tampered, cutoff))` membandingkan kedua frame sel demi sel dan melempar galat begitu satu sel berbeda.

**Q9.7** — CTR yang di-smooth = (3 + 20·0.10) / (10 + 20) = 5/30 ≈ 0.167. CTR mentahnya 3/10 = 0.30. Smoothing menarik estimasi lebih dari setengah jalan kembali ke 0.10 global karena item hanya memiliki 10 impresi dibandingkan kekuatan prior 20.

**Q9.8** — item_trend = (6/3 + 0.1) / (14/14 + 0.1) = 2.1 / 1.1 ≈ 1.91. Nilainya di atas 1, sehingga item tersebut sedang memanas: tiga hari terakhir rata-rata 2 klik per hari dibanding 1 klik per hari selama dua pekan terakhir.

**Q9.9** — Yang diharapkan: dengan jendela leak 1 hari, fitur yang bocor hanya mencakup sepertujuh jendela label, sehingga sebagian besar positif mendapat nilai nol dan kebocoran melemah; NDCG@20 dari ranker yang bocor seharusnya turun di bawah 0.5035 untuk leak 7 hari, dan gain share `leaky_future_clicks` juga seharusnya turun. Kemungkinan besar ia tetap jelas di atas 0.2116 dari model yang jujur, dan fitur tersebut tetap berperingkat dekat puncak. Bahkan sepotong kecil masa depan sudah cukup untuk membuat metrik offline tidak dapat dipercaya, itulah sebabnya satu-satunya aturan yang aman adalah cutoff keras tanpa pengecualian.

## Bab 10 — Learning to rank

**Q10.1** — B. Loss pointwise menilai setiap baris secara independen, loss pairwise membandingkan dua item dari pengguna yang sama, dan loss listwise melihat seluruh daftar berperingkat. Keluarga model (tree, jaringan saraf) ortogonal terhadap pembedaan ini, dan ketiganya dapat memakai label bergradasi maupun biner.

**Q10.2** — C. Setiap tree baru dicocokkan ke residual, yang untuk squared error sama dengan gradien negatif loss pada prediksi saat ini. Prediksi kemudian diperbarui sebesar $\eta$ kali nilai leaf tree tersebut.

**Q10.3** — D. Perubahan NDCG dari menukar sepasang item besar di dekat puncak dan untuk selisih gain yang besar, sehingga pasangan tersebut mendapat lambda besar. Perkalian ini tidak membuat NDCG dapat diturunkan; ia menggantikan gradien dengan gradien yang didefinisikan secara manual.

**Q10.4** — Argumen `group=` LightGBM adalah daftar ukuran grup yang berurutan, bukan id pengguna. Jika baris seorang pengguna tidak berdampingan, batas-batasnya memotong lintas pengguna, sehingga "query" mencampur baris pengguna yang berbeda dan pasangan serta NDCG dihitung atas daftar yang tidak bermakna. Biasanya tidak ada galat, hanya model yang lebih buruk, yang membuat bug ini sulit disadari.

**Q10.5** — Sedikit positif per daftar (sekitar seratus kandidat, sedikit aksi) memberi loss yang sadar-daftar sedikit sinyal tambahan dibanding label biner sederhana; label bergradasi (order) jarang dan berisik; dan batas atas candidate recall menekan perbedaan antara ranker yang baik. Selisihnya sekitar satu standar deviasi seed (0.2147 vs 0.2125 rata-rata ndcg@20, std 0.0014 dan 0.0021). Saya hanya akan berpindah jika tes offline dengan data lebih besar dan banyak seed, atau A/B test online, menunjukkan keunggulan yang konsisten.

**Q10.6** — `pred_contrib=True` mengembalikan satu kolom per fitur ditambah satu kolom bias terakhir (skor baseline). Mengiris `[:, :-1]` membuang bias sehingga hanya fitur yang tersisa. Tanpanya, kolom bias, konstanta besar seperti −2.9, sering menjadi nilai absolut terbesar dan akan dilaporkan sebagai "fitur" pada indeks di luar nama fitur, menyebabkan galat indeks atau penjelasan teratas yang palsu.

**Q10.7** — 0.393. Gain-nya 1, 0, 7; IDCG = 7·1 + 1·0.6309 + 0 = 7.631. Menukar posisi 1 dan 3: selisih gain 7 − 1 = 6, selisih diskon 1 − 0.5 = 0.5, sehingga perubahan DCG adalah 3.0 dan 3.0/7.631 = 0.393.

**Q10.8** — Sekitar +1.47. Kontribusi ditambah bias sama dengan skor, sehingga total kontribusi adalah 1.277 − (−2.917) = 4.194. Mengurangi delapan yang ditampilkan (2.726) menyisakan 1.468, yang dibulatkan menjadi +1.47 (±0.003 akibat pembulatan).

**Q10.9** — C. Penurunannya sekitar 1.6 standar deviasi noise, yang bersifat sugestif, bukan definitif. "Efek persis" yang tetap (A) mengabaikan noise seed, ambang 0.01 (B) bersifat sembarang, dan tanda negatif (D) persis seperti yang dihasilkan oleh grup yang berguna ketika dihapus.

**Q10.10** — Yang diharapkan (hipotesis, perlu diperiksa dengan menjalankannya): membuat order bernilai sama dengan klik menghilangkan hadiah ekstra untuk aksi terkuat, sehingga model mungkin meranking item yang dimasukkan keranjang dan dipesan kurang tajam dan `order_recall@20` bisa turun. Apakah ndcg@20 berubah lebih dari standar deviasi seed 0.0021 harus Anda nilai dari run Anda; perubahan yang lebih kecil adalah noise. Latihan ini mengajarkan bahwa `label_gain` menyandikan nilai bisnis dan bahwa perubahannya harus dinilai terhadap noise seed.

## Bab 11 — Re-ranking, diversitas, dan aturan bisnis

**Q11.1** — B. Dengan $\lambda = 1$, suku $(1-\lambda)\max\text{sim}$ nol, sehingga setiap langkah memilih relevansi tertinggi yang tersisa. Hasilnya adalah urutan asli; inilah sebabnya baris `MMR lambda=1.0` dalam hasil sama dengan baseline LambdaMART yang tidak dimodifikasi.

**Q11.2** — C. Membeli kembali kopi atau popok adalah perilaku normal yang seharusnya didorong oleh rekomender, sedangkan pengguna yang sudah memiliki kulkas tidak membutuhkan kulkas yang persis sama lagi. Aturan tersebut karenanya membuang item hanya ketika `ui_orders > 0` dan `item_is_consumable == 0`.

**Q11.3** — C. Thompson sampling menarik sampel dari posterior setiap item, sehingga item dengan sedikit data memiliki distribusi yang lebar dan kadang memenangkan slot, sedangkan item buruk yang terukur dengan baik hampir tidak pernah. Eksplorasi karenanya mengecil otomatis seiring bukti terkumpul, tanpa $\varepsilon$ tetap.

**Q11.4** — Aturan harus eksplisit, berbatas waktu, dan dapat diaudit, yang tidak bisa Anda jamin di dalam model yang dilatih. Jika impresi yang di-boost atau disponsori masuk ke data pelatihan sebagai bukti biasa, pelatihan ulang berikutnya akan membakar kampanye itu ke dalam model secara permanen dan mengacaukan estimasi relevansi. Dalam re-ranker, boost adalah suku yang terlihat dengan tanggal mulai dan akhir yang dapat dimatikan dan diuji tanpa pelatihan ulang.

**Q11.5** — Cap menghapus item dan tidak melakukan backfill, sehingga untuk pengguna yang pool kandidatnya terkonsentrasi di beberapa kategori atau toko, daftar yang tersisa bisa lebih pendek dari 50, dan deep recall turun. `recall@5` hampir tidak berubah karena beberapa item teratas jarang mengenai cap. (Ini penjelasan yang paling mungkin; skrip tidak mencatat panjang daftar.)

**Q11.6** — 8 item, dan cap per kategori yang bertanggung jawab. Ketiga puluh kandidat berbagi satu kategori, sehingga hanya yang posisinya dalam kategori di bawah `MAX_PER_CATEGORY = 8` yang bertahan; cap toko (paling banyak 3 item per toko di sini, di bawah 4) tidak menghapus apa pun, dan `head(20)` memiliki kurang dari 20 baris untuk dipertahankan.

**Q11.7** — Urutan: item 1, item 3, item 2. Langkah 1: skornya $0.5 \times$ rel $= 0.45, 0.40, 0.25$, sehingga item 1 (0.9) dipilih. Langkah 2: item 2 mendapat skor $0.5 \cdot 0.8 - 0.5 \cdot 1 = -0.10$ (kategori sama dengan item 1), item 3 mendapat skor $0.5 \cdot 0.5 - 0 = 0.25$, sehingga item 3 dipilih dan item 2 berada terakhir.

**Q11.8** — Beta(6, 16), rata-rata $6/22 \approx 0.273$, standar deviasi $\approx 0.093$. Variansnya $ab/((a+b)^2(a+b+1)) = 96/(484 \cdot 23) \approx 0.00862$. Item baru memiliki Beta(1, 1) dengan standar deviasi $\approx 0.289$, sekitar tiga kali lebih besar, sehingga sampelnya jauh lebih bervariasi dan ia akan sering menarik cukup tinggi untuk memenangkan slot sampai data mempersempitnya.

**Q11.9** — A. Hasil menunjukkan $-0.0035$ ndcg dan $+1.45$ kategori pada $\lambda = 0.9$, dan $-0.0715$ ndcg pada $\lambda = 0.3$. B salah (coverage@20 turun dari 0.4245 menjadi 0.3129), C salah (filter saja mengorbankan 0.0007), dan D salah (cap menaikkan kategori dari 5.33 menjadi 6.58).

**Q11.10** — Yang diharapkan (belum dijalankan di sini): dengan kategori saudara tidak lagi dihukum, item dari grup induk yang sama tidak lagi didorong turun, sehingga ndcg@20 pada $\lambda = 0.5$ seharusnya lebih tinggi daripada 0.1603 dalam tabel, sementara rata-rata kategori berbeda bisa tetap tinggi, karena kategori berbeda tetap dihitung per kategori. Halaman bisa saja terkonsentrasi pada lebih sedikit grup induk (misalnya hanya Pets). Bandingkan angka yang Anda ukur dengan prediksi ini dan dengan baris baseline.

## Bab 12 — Cold start, konteks, dan bias

**Q12.1** — B. `global_trending` tidak menerima argumen pengguna dan `generate` melakukan cross-join hasilnya ke setiap id pengguna, sehingga bahkan pengguna tanpa riwayat, atau `user_id = 0` anonim, menerima item yang trending secara global. Ia tidak berbasis sesi, ia berjalan pada setiap request bukan hanya saat gagal, dan ia meranking berdasarkan interaksi terbaru berbobot, bukan diskon.

**Q12.2** — C. Dengan $x = 0$ dan $n = 0$, rumus tereduksi menjadi $p_0 s / s = p_0$, CTR rata-rata katalog. Ini memperlakukan item baru sebagai rata-rata sampai ada bukti yang menyatakan sebaliknya, alih-alih sebagai nol atau tidak terdefinisi. Smoothing karenanya menghilangkan penalti karena tidak punya data.

**Q12.3** — C. Ranker memprediksi apa yang akan dilakukan pengguna dalam 7 hari ke depan, dengan label yang diagregasi per pekan. Jam request identik untuk setiap kandidat dalam grup seorang pengguna dan tidak memiliki label per-request untuk dipelajari. Konteks termasuk dalam model kedua yang real-time yang dilatih pada log impresi.

**Q12.4** — Ranker dilatih pada klik, sehingga item dengan sedikit lebih banyak klik mendapat skor lebih tinggi dan ditampilkan lebih sering. Exposure yang lebih banyak menghasilkan klik yang lebih banyak, yang melebarkan selisih, sementara item yang tidak pernah ditampilkan tidak pernah mendapat kesempatan membuktikan diri. Teknik yang mengurangi loop ini mencakup eksplorasi (epsilon-greedy atau Thompson sampling, menyisihkan slot untuk item yang belum terbukti), re-ranking diversitas, dan mencatat impresi beserta posisinya sehingga propensity weighting dapat mengoreksi exposure. Dua di antaranya sudah cukup.

**Q12.5** — Rekomender yang lambat merugikan pengguna sama seperti yang mati, sehingga timeout membatasi waktu tunggu terburuk hingga lima detik lalu menyajikan sesuatu yang berguna. JSON membawa `"fallback": true` dan `"model_version": "fallback"`. Frontend membaca flag tersebut dan menampilkan banner "Recommender offline".

**Q12.6** — Q berperingkat lebih tinggi. Query mengurutkan pertama berdasarkan jumlah event `order` dalam tujuh hari terakhir secara menurun (Q punya 1, P punya 0) dan hanya memakai `rating_count` untuk memecah seri. `i.id` terakhir membuat urutan deterministik ketika kedua kunci sebelumnya seri, sehingga request berulang mengembalikan daftar yang sama.

**Q12.7** — CTR mentah: A adalah $40/200 = 0.20$ dan B adalah $20/200 = 0.10$. Relevansi yang dikoreksi bias: A adalah $0.20/0.8 = 0.25$ dan B adalah $0.10/0.2 = 0.50$ (setara dengan setiap klik pada A berbobot 1.25 dan setiap klik pada B berbobot 5). B dua kali lebih relevan daripada A meskipun CTR mentahnya separuh. Angka mentah itu menyesatkan karena B ditampilkan di posisi yang jarang diperiksa pengguna.

**Q12.8** — Praktik. Harapkan bucket baru "0 clicks" berisi jauh lebih sedikit pengguna daripada bucket lain, sehingga metriknya berisik dan rasio `LambdaMART / pop_7d` di sana kurang bisa dipercaya dibanding rasio untuk bucket besar. Tanpa riwayat, ranker hanya bisa memakai kandidat trending dan fitur tingkat item, jadi perhatikan apakah keunggulannya atas `pop_7d` menyusut dibanding 5.22x pada bucket pengguna berat. Jika bucket-nya kosong, semua pengguna evaluasi skrip memiliki setidaknya satu klik sebelumnya dan barisnya tidak akan muncul. Intinya adalah kohort dingin itu kecil dan berisik.

**Q12.9** — B. Klik pada posisi dengan propensity yang sangat kecil mendapat bobot $1/\theta_k$, yang bisa berupa puluhan atau ratusan, sehingga satu klik yang beruntung bisa mendominasi pelatihan. Clipping membatasi bobot, menukar bias kecil dengan pengurangan variansi yang besar. Propensity adalah probabilitas dan tidak bisa negatif, bobot tidak perlu berjumlah satu, dan LightGBM menerima bobot besar.

## Bab 13 — Membangun sistemnya

**Q13.1** — B. Transaksi default read-only berarti setiap `INSERT`, `UPDATE`, atau `DELETE` dari rekomender gagal dengan `ReadOnlySqlTransaction`. Ini melindungi sistem of record dari bug dalam kode fitur atau model dan menjaga API Rust sebagai satu-satunya penulis. Ia tidak mempercepat pembacaan dan tidak ada hubungannya dengan gRPC.

**Q13.2** — A. Browser hanya boleh melaporkan `impression`, `click`, dan `hide`, yang merupakan pengamatan atas layar pembeli sendiri. `order` (dan `add_to_cart`) adalah konsekuensi dari aksi server-side yang terautentikasi, sehingga API menulisnya sendiri; menerimanya dari klien akan membiarkan siapa saja memalsukan sinyal positif kuat untuk item apa pun.

**Q13.3** — B. Listener `pagehide` milik tracker didaftarkan lebih dulu, sehingga ia mengosongkan antrean dan mengirim beacon sebelum listener halaman produk sendiri memasukkan klik ke antrean. Klik ditambahkan setelah kesempatan flush terakhir dan hilang. Perbaikannya adalah registri hook `onPageHide`: hook dijalankan dulu, lalu antrean dikosongkan dan di-beacon.

**Q13.4** — Id auto-increment ditetapkan saat insert tetapi baris menjadi terlihat saat commit. Jika transaksi yang memegang id 500 commit setelah yang memegang id 501, refresh yang sudah melihat 501 tidak akan pernah mengambil 500 dengan `id > last`. Membaca ulang 1000 id terakhir dan melakukan de-duplikasi berdasarkan `event_id` menangkap commit yang terlambat itu dengan biaya beberapa baris yang berlebih.

**Q13.5** — Setiap panggilan `Recommend` menyalin referensi ke booster dan versi saat ini ke variabel lokal di awal, sehingga ia mempertahankan pasangan model/versi yang konsisten selama seluruh durasinya. `retrain` membangun booster baru secara terpisah lalu menukar kedua atribut dengan penugasan referensi biasa. Request yang sedang berjalan selesai dengan model lama, request baru memakai yang baru, dan tidak diperlukan lock pada jalur baca.

**Q13.6** — Dua sesi. Event pukul 10:00 membuat sesi A; pada 10:20 sesi terakhir terlihat 20 menit lalu (di dalam jendela 30 menit), sehingga A dipakai ulang dan `last_seen_at`-nya menjadi 10:20. Pada 11:05 selisih sejak 10:20 adalah 45 menit, yang melebihi 30, sehingga sesi B baru disisipkan. Sesi pertama berakhir hari itu dengan `last_seen_at` = 10:20.

**Q13.7** — Sekitar 20.2% dari anggaran dan 3.985 detik. Pecahannya 1010.6 / 5000 = 0.2021. Staleness terburuknya 2 + 1 + 0.985 = 3.985 detik (delay flush + throttle refresh + latensi request).

**Q13.8** — Ia mengembalikan item aktif yang tersedia stoknya, diurutkan berdasarkan jumlah event `order` dalam 7 hari terakhir, lalu berdasarkan rating count. Item diberi label `source: "trending"` dengan skor 0 dan tanpa kontribusi karena berasal dari query popularitas non-personal dan tidak ada model yang menilainya; ini memungkinkan frontend memakai ulang rendering kartu yang sama. Respons juga membawa `fallback: true` dan `model_version: "fallback"` agar UI dapat menampilkan banner offline.

**Q13.9** — B. Protobuf mengidentifikasi field di wire berdasarkan nomor, bukan nama, sehingga dua sisi dengan penomoran berbeda untuk field yang sama tidak sepakat tentang byte mana yang berarti apa dan data terbuang atau salah dibaca tanpa galat kompilasi. Itulah sebabnya nomor field tidak boleh diubah atau dipakai ulang setelah pesan dipakai; hapus field dengan mencadangkan nomornya (reserve).

**Q13.10** — Harapkan latensi `p50` berubah sangat sedikit (semua run tetap di pita yang kira-kira sama dengan run k=20), dan baris kandidat-per-request identik di ketiga run, karena `candidates.generate` dan `features.build` tidak bergantung pada `k`; hanya pemotongan re-rank akhir dan jumlah baris SHAP yang berubah. Ini menunjukkan bahwa waktu didominasi oleh pekerjaan tetap per request (membangun riwayat dan fitur), bukan oleh penilaian atau ukuran respons.

## Bab 14 — Dari prototipe ke skala Shopee

**Q14.1** — B. Skew berarti fitur tersebut sah di kedua tempat tetapi dihitung secara berbeda (kode berbeda, filter berbeda seperti trafik bot), sehingga model belajar pada satu distribusi dan di-query dengan distribusi lain. Ukuran dataset, frekuensi retrain, dan aproksimasi ANN adalah isu yang terpisah. Feature store dengan satu definisi dan pencatatan fitur yang disajikan adalah pertahanan standar.

**Q14.2** — B. Local activation unit DIN membobot setiap perilaku masa lalu berdasarkan relevansinya terhadap kandidat yang sedang dinilai, sehingga vektor minat pengguna berubah menurut kandidat. Opsi A menggambarkan resep Embedding-and-MLP yang hendak diperbaiki oleh DIN. Opsi lainnya menggambarkan komponen yang tidak berkaitan.

**Q14.3** — A. Setiap task memiliki softmax gate sendiri atas pool expert bersama, sehingga task yang berkaitan dapat memilih expert yang mirip dan task yang bertentangan dapat mengkhususkan diri. Gate tidak mengubah dataset, menyamakan loss, atau menyimpan cache apa pun.

**Q14.4** — Pengguna merasakan bagian paling lambat dari sebuah request, dan halaman yang menyebar ke banyak layanan selambat layanan yang paling lambat, sehingga tail latency (p99) menentukan pengalaman; median menyembunyikan request yang lambat (di sini 985.0 ms pada p99 dibanding 839.9 ms pada p50). API Rust memberi panggilan gRPC timeout 5 detik; saat galat atau timeout ia mengembalikan daftar fallback berbasis popularitas dengan `fallback: true` alih-alih menggagalkan halaman.

**Q14.5** — Angka offline berasal dari log sistem sebelumnya, sehingga dapat digelembungkan oleh leakage (ranker yang bocor mendapat skor 0.5035 dibanding 0.2116 untuk yang jujur, dan akan tidak berguna secara live). Mereka juga mewarisi position bias dan mengabaikan pendapatan, diversitas, dan efek pada penjual. Alasan sah lainnya: selection bias dan feedback loop (model hanya dinilai pada item yang ditampilkan sistem lama). Hanya A/B test yang diacak yang mengukur efek sebenarnya.

**Q14.6** — Proporsi gabungan $\hat p = (5000+5150)/100000 = 0.1015$. SE $=\sqrt{0.1015 \times 0.8985 \times 2/50000} \approx 0.001910$. Selisihnya $0.1030 - 0.1000 = 0.0030$, sehingga $z = 0.0030/0.001910 \approx 1.57$. Nilai p dua sisi sekitar 0.116, yang di atas 0.05, sehingga peningkatan tersebut tidak signifikan secara statistik: data tidak dapat menyingkirkan efek nol, dan tes kemungkinan kurang bertenaga (underpowered).

**Q14.7** — Mentah: $2\times10^9 \times 128 \times 2 = 5.12\times10^{11}$ byte $= 512$ GB. Terkompresi: $2\times10^9 \times 32 = 6.4\times10^{10}$ byte $= 64$ GB. Product quantisation karenanya memberi pengurangan 8x di sini, dengan harga sebagian recall, yang disetel dengan jumlah probe dan re-scoring.

**Q14.8** — Dasar $= 1.0\times0.2 + 2.0\times0.05 + 3.0\times0.01 = 0.2 + 0.10 + 0.03 = 0.33$. Dengan boost, $0.33 \times (1 + 0.1) = 0.363$ (tanpa suku iklan). Dengan boost 0 skornya 0.33. Boost mengalikan seluruh campuran, sehingga sebuah kampanye mengangkat item secara proporsional terhadap seberapa baik model sudah menilainya.

**Q14.9** — B. Dengan jutaan pengguna, deviasi 0.6 poin dari 50/50 jauh di luar kebetulan dan menandakan sample ratio mismatch: bug pada pengacakan, pencatatan, penyaringan bot, atau redirect yang kehilangan salah satu arm. Hasil dari tes seperti itu tidak dapat dipercaya sampai penyebabnya ditemukan.

**Q14.10** — Harapkan baris $\lambda=0.8$ berada di antara baris 0.9 dan 0.7 pada kedua ukuran: ndcg@20 antara sekitar 0.2081 dan 0.1884, dan kategori berbeda yang lebih banyak daripada pada 0.9 tetapi lebih sedikit daripada pada 0.7 (antara sekitar 6.78 dan 11.03). Trade-off yang monoton mendukung klaim tersebut: setiap langkah menuju diversitas yang lebih besar mengorbankan relevansi, sehingga bobot diversitas adalah keputusan produk yang harus diuji secara online.
