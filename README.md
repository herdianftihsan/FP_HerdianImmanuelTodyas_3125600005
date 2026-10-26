# Final Project Oprec PENSHIP - Herdian Immanuel Todyas (3125600005)

## Identitas
- Nama: Herdian Immanuel Todyas
- NRP: 3125600005
- Program studi: 2 D4 Teknik Informatika

## Project Roboflow
- Project ID Roboflow: herdians-workspace-f6bfc/penship-detection-wbcjs, version 2
- Link Roboflow Universe: https://universe.roboflow.com/herdians-workspace-f6bfc/penship-detection-wbcjs

## Dataset
- Jumlah foto asli: bottle 97, laptop 44, pencil 129, background/jebakan 29
- Metode pengambilan foto: Pengambilan foto dilakukan secara mandiri menggunakan kamera HP dengan rasio 4:3. Variasi jarak mencakup *close-up*, sedang, dan jauh. Sudut mencakup *top-down*, sejajar, dan serong. Variasi pencahayaan meliputi kondisi terang lampu ruangan dan agak redup. Latar disesuaikan di atas ubin hijau dan meja. Terdapat juga skenario *occlusion* (botol/pensil terhalang tangan) serta penambahan foto *hard negatives* (tumbler, parfum, keyboard).
- Fitur AI labeling Roboflow yang digunakan (apabila ada) dan cara perapiannya:dilakukan *review* menyeluruh secara manual dengan merapatkan garis kotak ke tepi objek sesempit mungkin, serta menghapus *false positive* jika AI salah menandai objek lain.

## Training
- Lingkungan training: Laptop Lenovo IdeaPad Gaming 3 (NVIDIA GPU) berjalan di OS Linux Fedora
- Model: yolov8s (Small)
- Grafik: lihat folder grafik/

## Pengembangan Kode
Daftar perubahan terhadap kode dasar, beserta alasan dan pengaruhnya terhadap hasil:
1. **Implementasi `argparse` (CLI Arguments):** Menambahkan argumen `--model`, `--epochs`, dan `--batch` agar eksperimen model (nano vs small) dapat dieksekusi secara dinamis lewat terminal tanpa perlu melakukan *hardcode* di dalam skrip.
2. **Hyperparameter Tuning & Early Stopping:** Menggunakan optimizer `AdamW` dengan `lr0=0.001` untuk stabilitas konvergensi, serta menambahkan parameter `patience=20` agar training otomatis berhenti jika terdeteksi indikasi *overfitting*.
3. **Data Augmentation:** Menambahkan augmentasi `hsv_h`, `hsv_s`, `hsv_v`, `degrees=15.0`, `fliplr=0.5`, dan `mosaic=1.0` pada saat pemanggilan `model.train()`. Hal ini bertujuan agar model lebih tangguh (*robust*) terhadap perubahan intensitas cahaya silau/gelap serta rotasi objek (seperti pensil yang diletakkan miring).
4. **Auto-export Format ONNX:** Menambahkan fungsi `model.export(format="onnx")` di akhir siklus training agar *weights* model otomatis diekspor dan siap di-*deploy* ke *edge device* (seperti SBC Orange Pi di kapal).

## Hasil Eksperimen
| No | Skenario | Epochs | Batch | Optimizer | mAP50 | mAP50-95 | Keterangan |
|:--:|:---|:---:|:---:|:---:|:---:|:---:|:---|
| 0 | Baseline (kode dasar, yolov8n) | 50 | 16 | auto | - | - | Titik acuan |
| 1 | yolov8s + Augmentasi + AdamW | 80 | 16 | AdamW | 0.3737 | 0.14823 | Model terbaik untuk video |

## Analisis dan Kendala
- Analisis penyebab skenario terbaik menghasilkan nilai lebih tinggi: Skenario 1 (menggunakan `yolov8s.pt` dan augmentasi) menghasilkan mAP yang lebih tinggi karena kapasitas fitur model *Small* lebih besar dibanding *Nano*, sehingga mampu membedakan tekstur material *hard negatives* (misal: botol plastik transparan vs tumbler logam/parfum kaca) dengan lebih baik. Augmentasi rotasi (`degrees`) juga sangat membantu deteksi pensil kayu dalam berbagai posisi kemiringan.
- Kendala yang ditemui dan cara penyelesaiannya: Terdapat dua kendala utama. Pertama, *false positives* di mana AI mendeteksi *keyboard* sebagai pensil dan tumbler sebagai botol akibat *bias* bentuk. Hal ini diselesaikan dengan menyertakan foto *hard negatives* (meletakkan objek pengecoh bersebelahan dengan objek target) tanpa memberinya *bounding box*. Kedua, objek tipis (pensil biru/kuning di atas ubin hijau) sempat tidak terdeteksi dan mengalami *flickering* karena penyusutan piksel. Solusinya, saya melakukan *targeted data top-up* (menambah 20 foto *close-up* khusus pensil), merapatkan garis kotak *labeling*, dan mengeksekusi video prediksi menggunakan pergerakan kamera lambat (meminimalisir *motion blur*) dengan batas *confidence threshold* 0.3.