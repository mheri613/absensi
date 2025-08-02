"""
Kebutuhan library:
- requests
- scikit-learn
- matplotlib
- seaborn
- reportlab
Install dengan:
pip install requests scikit-learn matplotlib seaborn reportlab
"""
import os
import requests
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
import pickle

API_URL = "http://127.0.0.1:8000/api/attendance/verify-test"  # Ganti jika port/host berbeda
COURSE_ID = 1  # Ganti sesuai kebutuhan
dir_registered = os.path.join(os.path.dirname(__file__), "test_registered")
dir_unregistered = os.path.join(os.path.dirname(__file__), "test_unregistered")
CM_IMAGE = "confusion_matrix_test.png"
BAR_IMAGE = "bar_metrics.png"
PDF_OUTPUT = "hasil_pengujian_api_verify_test.pdf"

def test_api(image_path, course_id=COURSE_ID):
    with open(image_path, "rb") as img:
        files = {"image": img}
        params = {"course_id": course_id}
        try:
            r = requests.post(API_URL, files=files, params=params, timeout=300)
            try:
                data = r.json()
            except Exception as e:
                print(f"Error parsing JSON from {image_path}: {e}")
                print(f"Response content: {r.content}")
                return 0
            if data.get("success"):
                return 1  # Berhasil dikenali
            else:
                print(f"API returned error for {image_path}: {data.get('message')}")
                return 0  # Tidak dikenali
        except Exception as e:
            print(f"Error on {image_path}: {e}")
            return 0

def save_confusion_matrix_image(cm):
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Tidak Dikenali", "Dikenali"],
                yticklabels=["Tidak Dikenali", "Dikenali"])
    plt.xlabel("Prediksi")
    plt.ylabel("Label Sebenarnya")
    plt.title("Confusion Matrix API Verify-Test")
    plt.tight_layout()
    plt.savefig(CM_IMAGE)
    plt.close()

def save_bar_metrics(acc, prec, rec, f1):
    plt.figure(figsize=(6,4))
    metrics = [acc, prec, rec, f1]
    labels = ["Akurasi", "Precision", "Recall", "F1-Score"]
    sns.barplot(x=labels, y=metrics, palette="Blues_d")
    plt.ylim(0, 1)
    plt.ylabel("Nilai")
    plt.title("Visualisasi Metrik Evaluasi")
    for i, v in enumerate(metrics):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(BAR_IMAGE)
    plt.close()

def generate_pdf_report(cm, acc, prec, rec, f1, results):
    doc = SimpleDocTemplate(PDF_OUTPUT, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()
    title = Paragraph("Laporan Hasil Pengujian API /attendance/verify-test", styles['Title'])
    elements.append(title)
    elements.append(Spacer(1, 12))

    # Penjelasan confusion matrix
    penjelasan_cm = (
        "<b>Penjelasan Confusion Matrix:</b><br/>"
        "Confusion matrix adalah tabel evaluasi yang menunjukkan jumlah prediksi benar dan salah dari sistem.<br/>"
        "<b>Baris:</b> label sebenarnya (ground truth).<br/>"
        "<b>Kolom:</b> hasil prediksi sistem.<br/>"
        "<ul>"
        "<li><b>True Positive (TP):</b> Wajah terdaftar dan dikenali dengan benar.</li>"
        "<li><b>False Positive (FP):</b> Wajah tidak terdaftar tapi sistem salah mengenali.</li>"
        "<li><b>True Negative (TN):</b> Wajah tidak terdaftar dan sistem juga tidak mengenali.</li>"
        "<li><b>False Negative (FN):</b> Wajah terdaftar tapi gagal dikenali.</li>"
        "</ul>"
    )
    elements.append(Paragraph(penjelasan_cm, styles['Normal']))
    elements.append(Spacer(1, 12))

    # Penjelasan metrik
    penjelasan_metrik = (
        "<b>Penjelasan Metrik Evaluasi:</b><br/>"
        "<ul>"
        "<li><b>Akurasi:</b> Persentase prediksi yang benar dari seluruh data uji.</li>"
        "<li><b>Precision:</b> Ketepatan sistem dalam mengenali wajah yang benar-benar terdaftar.</li>"
        "<li><b>Recall:</b> Kemampuan sistem mendeteksi semua wajah yang seharusnya dikenali.</li>"
        "<li><b>F1-Score:</b> Rata-rata harmonis precision dan recall.</li>"
        "</ul>"
    )
    elements.append(Paragraph(penjelasan_metrik, styles['Normal']))
    elements.append(Spacer(1, 12))

    # Metrik
    metrics = [
        ["Akurasi", f"{acc:.2f}"],
        ["Precision", f"{prec:.2f}"],
        ["Recall", f"{rec:.2f}"],
        ["F1-Score", f"{f1:.2f}"]
    ]
    elements.append(Paragraph("<b>Metrik Evaluasi:</b>", styles['Heading2']))
    t = Table(metrics, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold')
    ]))
    elements.append(t)
    elements.append(Spacer(1, 12))

    # Confusion Matrix
    elements.append(Paragraph("<b>Confusion Matrix:</b>", styles['Heading2']))
    cm_data = [["", "Pred: Tidak Dikenali", "Pred: Dikenali"],
               ["Label: Tidak Dikenali", cm[0,0], cm[0,1]],
               ["Label: Dikenali", cm[1,0], cm[1,1]]]
    t2 = Table(cm_data, hAlign='LEFT')
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold')
    ]))
    elements.append(t2)
    elements.append(Spacer(1, 12))

    # Gambar confusion matrix
    if os.path.exists(CM_IMAGE):
        elements.append(Paragraph("<b>Visualisasi Confusion Matrix:</b>", styles['Heading2']))
        elements.append(RLImage(CM_IMAGE, width=300, height=240))
        elements.append(Spacer(1, 12))

    # Gambar bar chart metrik
    if os.path.exists(BAR_IMAGE):
        elements.append(Paragraph("<b>Visualisasi Metrik Evaluasi:</b>", styles['Heading2']))
        elements.append(RLImage(BAR_IMAGE, width=350, height=250))
        elements.append(Spacer(1, 12))

    # Tabel hasil prediksi
    elements.append(Paragraph("<b>Detail Hasil Prediksi:</b>", styles['Heading2']))
    table_data = [["Nama File", "Label Sebenarnya", "Prediksi"]]
    for fname, label, pred in results:
        table_data.append([fname, "Dikenali" if label==1 else "Tidak Dikenali", "Dikenali" if pred==1 else "Tidak Dikenali"])
    t3 = Table(table_data, hAlign='LEFT')
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold')
    ]))
    elements.append(t3)

    doc.build(elements)
    print(f"\nLaporan PDF berhasil dibuat: {PDF_OUTPUT}")

def main():
    y_true, y_pred, results = [], [], []

    # Uji registered
    for fname in os.listdir(dir_registered):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(dir_registered, fname)
            pred = test_api(img_path)
            y_true.append(1)
            y_pred.append(pred)
            results.append((fname, 1, pred))
            print(f"{fname}: True=1, Pred={pred}")

    # Uji unregistered
    for fname in os.listdir(dir_unregistered):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(dir_unregistered, fname)
            pred = test_api(img_path)
            y_true.append(0)
            y_pred.append(pred)
            results.append((fname, 0, pred))
            print(f"{fname}: True=0, Pred={pred}")

    # Evaluasi
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("\n=== Hasil Pengujian API Verify-Test ===")
    print("Confusion Matrix:\n", cm)
    print(f"Accuracy : {acc:.2f}")
    print(f"Precision: {prec:.2f}")
    print(f"Recall   : {rec:.2f}")
    print(f"F1-Score : {f1:.2f}")

    # Simpan gambar confusion matrix dan bar chart metrik
    save_confusion_matrix_image(cm)
    save_bar_metrics(acc, prec, rec, f1)

    # Generate PDF
    generate_pdf_report(cm, acc, prec, rec, f1, results)

if __name__ == "__main__":
    main()