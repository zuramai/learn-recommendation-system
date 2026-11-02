# Kata Pengantar

Buka aplikasi Shopee pada Selasa malam. Sebelum Anda mengetik satu huruf pun, layar beranda sudah menampilkan empat puluh produk: casing untuk ponsel yang Anda beli bulan lalu, makanan kucing karena minggu lalu Anda menjelajahi produk hewan peliharaan, dan sebuah ketel listrik *flash sale*. Tidak ada orang yang menyusun slot-slot itu dengan tangan. Sebuah *pipeline* menelusuri jutaan produk, membuang hampir semuanya dalam hitungan milidetik, menilai yang tersisa, lalu mengacak urutan para pemenangnya agar halaman tidak tampak seperti tembok berisi casing ponsel yang seragam.

*Pipeline* itu adalah **sistem rekomendasi** (*recommender system*). Buku ini mengajarkan cara kerjanya dengan membangun satu: mula-mula daftar item paling populer, lalu, bab demi bab, setiap gagasan yang mengubah daftar itu menjadi sistem multi-tahap modern.

## Gambaran besar dengan bahasa sederhana

Sebuah recommender menjawab satu pertanyaan: **"Dari semua yang kami jual, beberapa item mana yang sebaiknya dilihat orang ini sekarang?"** Ia memakai petunjuk yang sudah dimiliki toko: apa yang pernah diklik dan dibeli orang itu, serta apa yang dilakukan orang-orang yang mirip.

Kesulitannya ada pada ukuran. Marketplace seperti milik kita punya ribuan item (jutaan di aplikasi sungguhan), sedangkan layar hanya muat sekitar empat puluh. Kita tidak mungkin menilai setiap item dengan cermat untuk setiap pengunjung dalam sepersekian detik sebelum halaman tampil. Maka sistem sungguhan bekerja seperti pramuniaga yang mengambilkan barang dari gudang raksasa untuk seorang pelanggan:

1. **Retrieval (pembangkitan kandidat): ambil satu troli penuh dari gudang.** Pramuniaga tidak memeriksa setiap kardus. Ia berjalan ke lorong hewan peliharaan dan lorong aksesori ponsel, karena riwayat Sari menunjukkan ia menyukainya, lalu mengambil beberapa ratus kardus dengan cepat. Langkah ini harus *cepat*, dan boleh agak kasar. Beberapa ratus item yang dihasilkannya adalah **kandidat** (*candidate*).
2. **Ranking (pemeringkatan): periksa dengan teliti dan urutkan isi troli.** Kini pramuniaga hanya memegang beberapa ratus kardus, sehingga ia sanggup memeriksa satu per satu: seberapa besar kemungkinan Sari mengeklik ini, mengingat rentang harganya, pesanan sebelumnya, dan jam saat ini? Setiap kandidat mendapat skor dan daftar diurutkan dari terbaik ke terburuk. Langkah ini *presisi* tetapi lambat per item, itulah sebabnya ia hanya melihat kandidat.
3. **Re-ranking (pemeringkatan ulang): menata etalase.** Jika diurutkan hanya berdasarkan skor, sepuluh teratas bisa berupa sepuluh casing ponsel yang nyaris sama. Pramuniaga menata ulang agar bervariasi, menurunkan produk yang stoknya habis, dan mengikuti aturan toko, misalnya tidak menampilkan penjual yang sama lima kali berturut-turut. Sentuhan akhir ini disebut **re-ranking**.

```mermaid
flowchart LR
    W["Semua item<br/>(gudang)"] -->|"1. Retrieval<br/>cepat dan kasar"| C["Beberapa ratus<br/>kandidat"]
    C -->|"2. Ranking<br/>lambat dan presisi"| R["Daftar berskor<br/>dan terurut"]
    R -->|"3. Re-ranking<br/>variasi dan aturan"| P["Sekitar 40 item<br/>yang dilihat Sari"]
```

*Gambar: Corong tiga tahap. Setiap tahap melihat lebih sedikit item daripada tahap sebelumnya, sehingga sanggup bekerja lebih banyak per item.*

Corong ini ada karena sebuah pertukaran: penilaian yang cermat memakan waktu, sedangkan waktu terbatas, jadi kita menghabiskan usaha murah untuk banyak item dan usaha mahal untuk sedikit item. Tanpa corong, Anda akan menampilkan halaman yang lambat (menilai semuanya dengan cermat) atau rekomendasi yang buruk (menilai semuanya dengan kasar).

Kita membangun bentuk persis ini selangkah demi selangkah. Pertama, di Bab 3 sampai 7, satu metode mencoba mengerjakan seluruh tugas. Lalu Bab 8 sampai 12 membagi pekerjaan menjadi tiga tahap di atas.

## Buku ini untuk siapa

Buku ini ditulis untuk **software engineer tingkat menengah**: Anda nyaman membaca Python, mengenal REST dan SQL, dan pernah merilis kode ke produksi. Latar belakang *machine learning* tidak diperlukan; setiap bagian matematika dijelaskan saat pertama kali muncul dan dipakai pada contoh kecil yang dihitung dengan tangan. Semuanya berjalan di laptop dalam hitungan menit, tanpa GPU.

Dua janji membentuk seluruh isi buku ini:

1. **Setiap algoritma dijelaskan selangkah demi selangkah.** Anda akan melihat intuisi, rumus, contoh hitungan, lalu kode sungguhan yang mengimplementasikannya.
2. **Setiap angka dalam teks berasal dari program yang bisa Anda jalankan.** Tidak ada hasil dalam buku ini yang diketik tangan. Setiap bab punya skrip pendamping; keluarannya ditempel ke buku langsung dari berkas yang ditangkap saat skrip dijalankan. Jika Anda menjalankan ulang skripnya, Anda akan mendapat angka yang sama, karena semua *random seed* dikunci. *Random seed* ibarat melempar dadu dalam permainan di mana semua orang mendapat angka yang sama dengan urutan yang sama: bagian "acak" pada simulasi dan pelatihan menghasilkan hasil identik setiap kali dijalankan, sehingga angka Anda cocok dengan angka di buku.

## Bentuk buku ini

Buku ini mengikuti jalur historis dan praktis bidangnya. Setiap bab menambahkan perangkat baru hanya setelah pendekatan sebelumnya menunjukkan batasnya pada data yang sama.

- **Fondasi (Bab 1–2).** Masalahnya, datanya, dan penggarisnya: precision, recall, NDCG, dan evaluasi berbasis waktu. Singkatnya, tiga skor untuk menjawab "seberapa bagus daftar ini?" dan kebiasaan menguji recommender pada hari-hari yang belum pernah dilihatnya.
- **Recommender model tunggal (Bab 3–7).** Satu metode mengerjakan seluruh tugas. Popularitas (tampilkan yang disukai semua orang), *content-based filtering* (sarankan item yang deskripsinya mirip dengan yang Anda sukai), *collaborative filtering* (sarankan yang disukai orang-orang yang mirip dengan Anda), faktorisasi matriks (memampatkan selera semua orang menjadi beberapa angka tersembunyi), serta *embedding* dengan *retrieval* dua menara (*two-tower*) (mengubah pengguna dan item menjadi deretan angka sehingga "dekat" berarti "cocok"). Setiap istilah dijelaskan dari nol di babnya masing-masing.
- **Sistem multi-tahap (Bab 8–12).** Marketplace sungguhan menjalankan sebuah corong (*funnel*), yang dijelaskan di bagian berikutnya: tahap murah mengambil beberapa ratus **kandidat** (*candidate*), tahap presisi **memeringkat** (*rank*) kandidat itu, dan tahap akhir **memeringkat ulang** (*re-rank*) demi keberagaman dan aturan bisnis. Bab-bab ini membahas pembangkitan kandidat, rekayasa fitur (mengubah log mentah menjadi angka yang berguna bagi model), *learning to rank*, *re-ranking*, serta *cold start*, konteks, dan bias.
- **Produksi (Bab 13–14).** Sebuah aplikasi Vue (halaman web), Rust (server), Postgres (basis data), dan Python gRPC yang berjalan, tempat Anda mengikuti satu klik dari ujung ke ujung, lalu apa yang berubah pada puluhan juta pengguna.

Lampiran berisi jawaban kuis, glosarium, perintah reproduksi, dan daftar pustaka.

## Isi repositori

Buku ini berada di dalam repositori sebuah proyek lengkap yang berjalan. Repositori itu berisi buku, kode yang diajarkan buku, dan aplikasi yang memakainya.

```mermaid
flowchart LR
    subgraph repo["recommendation-system/"]
        proto["proto/<br/>recommender.proto<br/>(kontrak gRPC)"]
        backend["backend/<br/>API Rust + Axum<br/>(menulis event)"]
        recsys["recommender/<br/>paket Python 'recsys'<br/>(layanan gRPC, membaca DB)"]
        frontend["frontend/<br/>etalase Vue 3"]
        book["book/<br/>buku ini:<br/>bab, kode, hasil"]
        docs["docs/<br/>GUIDE.md"]
    end
    frontend -->|"HTTP /api"| backend
    backend -->|"gRPC"| recsys
    proto -.->|"menghasilkan klien"| backend
    proto -.->|"menghasilkan server"| recsys
    backend --> pg[("PostgreSQL")]
    recsys -->|"hanya-baca"| pg
    book -.->|"mengimpor dan menjalankan"| recsys
    book -.->|"menyertakan kode dari"| backend
    docs -.->|"menjelaskan"| repo
```

*Gambar: Gambaran repositori sekilas. Panah penuh adalah pemanggilan saat runtime; panah putus-putus adalah hubungan saat build atau dokumentasi.*

Setiap direktori punya satu tugas:

| Direktori | Isinya | Kemunculannya dalam buku |
|---|---|---|
| `recommender/` | Paket Python `recsys`: simulator marketplace, *retriever* kandidat, pembangun fitur, ranker LightGBM, re-ranker, dan layanan gRPC | Bab 1–3 dan 8–13 |
| `book/code/` | Satu skrip pendamping per bab algoritma, ditambah `common.py` yang dipakai bersama | Setiap bab |
| `book/results/` | Keluaran (`.txt`) dan metrik (`.json`) hasil tangkapan skrip-skrip itu | Setiap blok `output` |
| `backend/` | API Rust dan Axum: autentikasi, katalog, keranjang, *checkout*, pencatatan event, klien gRPC | Bab 13 |
| `frontend/` | Etalase Vue 3 dengan penjelasan "Why this?" dan log event langsung | Bab 13 |
| `proto/` | `recommender.proto`, satu berkas yang mendefinisikan kontrak antara Rust dan Python | Bab 13 |
| `docs/` | `GUIDE.md`, panduan berbahasa sederhana tentang aplikasi yang berjalan | Bab 13 |
| `book/` | Bab, grafik, tangkapan layar, dan perangkat build untuk PDF yang sedang Anda baca | – |

Dua keputusan desain muncul berulang kali. **Rust menulis semua perilaku pengguna ke Postgres; Python hanya membacanya**, lewat koneksi hanya-baca (koneksi yang bisa melihat data tetapi tidak bisa mengubahnya, sehingga recommender tidak akan pernah merusak catatan toko). Dan **rekomendasi dihitung saat diminta**, artinya daftar dibangun pada saat Anda membuka halaman, sehingga klik lima detik lalu sudah bisa mengubah beranda, sementara model (bagian yang dipelajari) dilatih ulang secara berkala di latar belakang.

Beberapa istilah pada tabel di atas, dengan bahasa sederhana. **gRPC** adalah cara agar dua program, di sini program Rust dan program Python, saling memanggil fungsi lewat jaringan. **Berkas `.proto`** adalah kesepakatan tertulis tentang panggilan apa saja yang ada dan apa yang dibawanya. **LightGBM** adalah pustaka yang belajar menilai item; Bab 10 menjelaskannya.

## Persiapan

Anda bisa membaca buku ini tanpa menjalankan apa pun, tetapi Anda akan belajar lebih banyak jika mengubah satu angka dan melihat apa yang terjadi.

### Yang Anda butuhkan

| Perangkat | Kegunaan |
|---|---|
| [`uv`](https://docs.astral.sh/uv/) | Lingkungan Python untuk `recommender/` dan `book/` |
| Node.js dan `npm` | Frontend Vue, serta pustaka `mermaid` dan `katex` yang dipakai merender diagram dan matematika di PDF |
| PostgreSQL | Hanya untuk aplikasi langsung di Bab 13. Berjalan di `localhost:5432`, user `postgres`, password `postgres` |
| Rust toolchain (`cargo`) | Hanya untuk backend |
| `protoc` | Hanya untuk mengompilasi kontrak gRPC bagi Rust |

Bab algoritma (1–12) **hanya membutuhkan `uv`**. Postgres, Rust, dan Node baru diperlukan saat Anda ingin menjalankan aplikasi web lengkap.

### Skrip pendamping dan perangkat build buku

*Clone* repositori, lalu dari root repositori jalankan:

```bash
cd book && uv sync && uv run playwright install chromium && npm i
```

`uv sync` memasang dependensi Python, termasuk paket `recsys` dari `../recommender` dalam mode *editable*, sehingga skrip mengimpor kode yang sama dengan yang ditampilkan bab. Langkah `playwright install` mengunduh Chromium *headless*, yang dipakai untuk merender PDF sekaligus mengambil tangkapan layar. `npm i` mengunduh `mermaid` dan `katex`.

Setiap skrip pendamping dijalankan dari direktori `book/`:

```bash
uv run python code/ch01_explore.py
```

Untuk membuat ulang semua berkas hasil secara berurutan, gunakan `uv run python tools/run_all.py`. Lampiran C mencantumkan semua perintah, termasuk yang membangun ulang grafik dan PDF.

### Aplikasi web

Untuk Bab 13 Anda memerlukan ketiga layanan berjalan. Petunjuk langkah demi langkah yang otoritatif ada di `docs/GUIDE.md`, bagian "Run it". Garis besarnya:

1. Buat basis data dan terapkan skemanya: `createdb marketplace`, salin `backend/.env.example` ke `backend/.env`, lalu `cargo run -- migrate` di dalam `backend/`.
2. Isi basis data dengan marketplace simulasi: `uv run recsys seed-db --yes` di dalam `recommender/`.
3. Buka tiga terminal: `uv run recsys serve` (Python, port 50051), `cargo run` (Rust, port 8080), dan `npm run dev` (Vue, port 5173).

Lalu buka `http://localhost:5173`. Data seed menyertakan akun demo seperti `user43@demo.test` dengan password `password`; halaman login mencantumkan akun lainnya.

## Contoh berjalan: marketplace simulasi

Log marketplace asli bersifat privat, dan dataset publik tidak memuat posisi (di mana item ditampilkan di halaman), durasi tinggal (*dwell time*, berapa lama pengguna bertahan di item itu), dan stok, sehingga buku ini memakai **marketplace simulasi** yang disertakan dalam repositori: toko pura-pura kecil yang dijalankan program, dengan pembeli pura-pura yang menjelajah dan membeli. Karena kita sendiri yang menulis simulatornya, kita juga tahu kebenaran tersembunyi di balik perilaku itu, yang dipakai Bab 1 untuk menunjukkan bagaimana data log memelintir apa yang sebenarnya diinginkan pengguna.

Simulasi bawaan, yang dikonfigurasi di `recommender/recsys/simulate.py`, memiliki 1,500 pengguna, 2,500 item, 150 toko, dan 60 hari aktivitas, dengan *seed* tetap. Skrip Bab 1 mengonfirmasi ukuran tabelnya:

```output
file: results/ch01_explore.txt
lines: 2-9
```

Selain ukurannya, simulator mereproduksi pola-pola yang membuat rekomendasi menarik. Beberapa istilah di bawah ini baru, jadi masing-masing disertai arti sederhananya:

- **Popularitas timpang.** Sebagian kecil item menarik sebagian besar interaksi, persis seperti di toko sungguhan. Jika item diurutkan dari yang paling populer ke yang paling sepi, Anda mendapat beberapa batang tinggi diikuti ekor yang sangat panjang dan rendah; karena itu disebut **long tail** (ekor panjang). Bahayanya, recommender yang hanya mengulang batang-batang tinggi tidak pernah menunjukkan sisa rak kepada pembeli.
- **Selera bersifat personal.** Setiap pengguna punya preferensi tersembunyi terhadap kategori, rentang harga, dan merek. Sari mungkin menyukai makanan kucing dan casing ponsel murah; pengguna lain mungkin lebih suka elektronik premium.
- **Waktu berpengaruh.** Lalu lintas lebih tinggi pada akhir pekan dan sekitar tanggal gajian (tanggal 25 hingga tanggal 1 setiap bulan), dan jendela simulasi memuat **mega-sale 9.9**, hari saat lalu lintas dan pesanan melonjak. Bab 1 mengukur dan membuat grafik efek-efek ini.
- **Perilaku berbentuk corong.** Setiap tindakan dicatat sebagai *event* (kejadian). Pengguna melihat sebuah item di layar (sebuah **impression**), sebagian impression menjadi **click** (klik), lebih sedikit menjadi **add-to-cart** (item dimasukkan ke keranjang), dan lebih sedikit lagi menjadi **order** (pembelian). Setiap langkah melibatkan lebih sedikit orang daripada langkah sebelumnya, seperti corong yang menyempit. Beberapa item di-*hide* (pengguna menekan "tidak tertarik") dan beberapa pembelian di-*return* (dikembalikan).
- **Stok habis dan item baru muncul.** Penjual mendaftarkan produk selama jendela simulasi dan sebagian terjual habis. Item yang benar-benar baru dan belum pernah diklik siapa pun sulit direkomendasikan karena belum ada perilaku untuk dipelajari; inilah **masalah cold start**, bersama masalah ketersediaan, di Bab 12.

### Satu pembagian waktu untuk seluruh buku

Konvensi terpenting dalam buku ini adalah satu titik waktu tetap bernama **cutoff**. Marketplace simulasi mencakup 60 hari. Kita memilih **hari ke-53 sebagai cutoff** dan menerapkan satu aturan di mana-mana:

> Setiap algoritma boleh belajar dari, dan menghitung fitur dari, event yang terjadi **sebelum** cutoff. Algoritma itu lalu dinilai berdasarkan apa yang benar-benar dilakukan pengguna dalam **7 hari setelahnya**.

Bayangkan seorang peramal cuaca. Anda memberinya catatan cuaca sampai 1 Maret dan bertanya, "Bagaimana cuaca 2 sampai 8 Maret?" Anda sebenarnya sudah tahu jawabannya karena ada di catatan Anda, tetapi Anda menyembunyikannya, lalu membandingkan ramalan dengan kenyataan. Perbandingan itulah skornya. *Curang* berarti membiarkan peramal membaca koran 5 Maret lebih dulu. Di marketplace kita, **riwayat** (*history*) adalah catatannya, cutoff adalah "1 Maret", dan **jendela label** (*label window*, hari-hari setelah cutoff) adalah kunci jawaban yang disembunyikan. Aturan yang sama berlaku bagi pembeli kita, Sari: recommender hanya melihat apa yang Sari lakukan sebelum hari ke-53, lalu diperiksa terhadap apa yang ia lakukan pada hari ke-53 sampai 60. Kalau recommender boleh melihat pembelian Sari di hari ke-55 saat belajar, skornya tampak bagus tetapi tidak bermakna.

```mermaid
flowchart LR
    A["Hari 0<br/>simulasi dimulai"] --> B["Hari 0 sampai 53<br/>RIWAYAT<br/>algoritma belajar di sini"]
    B --> C["Hari 53<br/>CUTOFF"]
    C --> D["Hari 53 sampai 60<br/>JENDELA LABEL<br/>apa yang benar-benar dilakukan pengguna"]
    D --> E["Hari 60<br/>simulasi berakhir"]
```

*Gambar: Pembagian waktu yang dipakai setiap skrip pendamping. Tidak ada data dari jendela label yang pernah terlihat saat pelatihan atau pembangunan fitur.*

Berkas bersama `book/code/common.py` menegakkan pembagian ini dan menyediakan satu fungsi `evaluate`, sehingga angka-angka dapat dibandingkan antarbab.

Anggaplah `evaluate` sebagai kunci jawaban tunggal yang dipakai untuk menilai setiap algoritma. Jika tiap bab memakai skema penilaian sendiri, Anda tidak bisa tahu apakah sebuah gagasan baru memang lebih baik atau hanya dinilai lebih longgar.

## Bagaimana setiap bab disusun

Bab 1 sampai 14 mengikuti templat yang sama, sehingga Anda selalu tahu posisi Anda:

1. **Cerita pembuka** dengan skenario ala Shopee, dan daftar singkat hal yang akan Anda pelajari.
2. **Intuisi**, memakai analogi sehari-hari sebelum rumus apa pun.
3. **Algoritma**: rumus, langkah bernomor, dan contoh hitungan yang cukup kecil untuk dikerjakan dengan pensil.
4. **Penelusuran kode**: kode sungguhan dari repositori, dijelaskan bagian demi bagian.
5. **Jalankan**: perintah untuk dieksekusi beserta keluaran hasil tangkapannya.
6. **Hasil**: grafik dan interpretasi yang terkait dengan angka sebenarnya.
7. **Jebakan dan praktik produksi**: apa yang bisa salah, dan bagaimana sistem besar menanganinya.
8. **Ringkasan** dan **Kuis**.

### Konvensi visual dan tipografi

Kode diambil langsung dari berkas repositori, dan blok keluaran disalin dari berkas hasil yang ditangkap. Setiap angka dalam paragraf juga muncul di blok keluaran atau berkas JSON di bawah `book/results/`. Matematika sebaris tampak seperti $a \cdot b$; Bab 2 membangun NDCG dari nol. Angka ditulis dengan titik desimal, sama seperti pada keluaran program.

Tiga skor kualitas muncul mulai Bab 2. Anda belum memerlukannya sekarang, tetapi berikut arti satu kalimat masing-masing agar terasa akrab. **Precision** bertanya: dari item yang kami tampilkan, berapa bagian yang benar-benar diinginkan pengguna? **Recall** bertanya: dari semua item yang benar-benar diinginkan pengguna, berapa bagian yang berhasil kami tampilkan? **NDCG** (*Normalised Discounted Cumulative Gain*) adalah satu skor yang memberi ganjaran bila item yang tepat berada di *puncak* daftar, bukan di dasar. Bab 2 membangun masing-masing dengan contoh kecil.

Empat jenis kotak callout menyela teks. Tampilannya di PDF seperti berikut:

**Kotak "Tahukah kamu?"** (biru) memuat fakta yang dapat diverifikasi dari literatur riset atau sejarah bidang ini, dengan sumbernya disebut di bagian bawah kotak dan tercantum dalam daftar pustaka. Setiap bab memiliki sedikitnya dua.

::: tip
**Tip** adalah pintasan praktis atau kebiasaan yang akan menghemat waktu Anda.
:::

::: warning
**Peringatan** menandai kesalahan yang mudah dibuat dan mahal untuk ditemukan, seperti mengevaluasi pada data yang sudah pernah dilihat model.
:::

::: try
**Kotak "Coba sendiri"** menyarankan eksperimen kecil: ubah satu parameter di sebuah skrip, jalankan ulang, dan bandingkan. Inilah cara terbaik mengubah bacaan menjadi pemahaman.
:::

Terakhir, setiap bab ditutup dengan **kuis** berisi delapan sampai sepuluh soal: pilihan ganda, jawaban singkat, satu soal membaca kode, satu soal hitungan, dan satu latihan praktik yang memodifikasi skrip pendamping. Jawaban beserta penjelasan singkat ada di Lampiran A. Cobalah menjawab sebelum melihatnya; usaha mengingat kembali sebuah jawaban inilah yang membuatnya melekat.

## Catatan tentang skala

Saat kami menjelaskan cara kerja platform besar, kami mengandalkan makalah yang telah dipublikasikan dan tidak mengarang angka internal. Ketika metode sederhana mengalahkan yang lebih canggih pada dataset ini, buku ini mengatakannya dan menjelaskan alasannya.

Dengan peta di tangan, bukalah halaman berikutnya. Bab 1 dimulai dari tempat setiap proyek rekomendasi dimulai: dari data, dan dari tinjauan jujur tentang apa yang bisa dan tidak bisa dikatakannya kepada kita.
