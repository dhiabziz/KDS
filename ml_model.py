import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss
from sklearn.multioutput import MultiOutputClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Fungsi untuk ekstraksi fitur k-mer dari urutan DNA
def extract_kmer_features(sequence, k=3):
    """
    Ekstraksi fitur k-mer dari urutan DNA
    
    Args:
        sequence (str): Urutan DNA
        k (int): Panjang k-mer
        
    Returns:
        dict: Frekuensi k-mer
    """
    kmers = {}
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        if kmer in kmers:
            kmers[kmer] += 1
        else:
            kmers[kmer] = 1
    return kmers

# Fungsi untuk memuat dan memproses dataset
def load_and_process_data():
    """
    Memuat dan memproses dataset
    
    Returns:
        tuple: X_train, X_test, y_train, y_test, label_info, data
    """
    print("Memuat dataset...")
    
    # Memuat dataset
    try:
        gene_seq = pd.read_csv('gene_seq.csv')
        gene_go = pd.read_csv('gene_data_GO.csv')
        gene_locus = pd.read_csv('gene_data_locus.csv')
        
        print(f"Dataset berhasil dimuat: {len(gene_seq)} sekuens, {len(gene_go)} anotasi GO, {len(gene_locus)} locus")
    except FileNotFoundError:
        print("Dataset tidak ditemukan. Menggunakan data dummy untuk demonstrasi...")
        
        # Membuat data dummy jika file tidak ditemukan
        np.random.seed(42)
        
        # Dummy gene_seq
        bases = ['A', 'T', 'G', 'C']
        gene_seq = pd.DataFrame({
            'sysname': [f'YAL{i:03d}W' for i in range(100)],
            'gene': [''.join(np.random.choice(bases, 100)) for _ in range(100)],
            'promoter': [''.join(np.random.choice(bases, 50)) for _ in range(100)],
            'protein': [''.join(np.random.choice(['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I'], 30)) for _ in range(100)]
        })
        
        # Dummy gene_go
        go_terms_bp = ['mitophagy', 'mitochondrion organization', 'biological_process unknown', 'phospholipid homeostasis']
        go_terms_cc = ['mitochondrion', 'mitochondrial outer membrane', 'membrane', 'integral component of membrane']
        go_terms_mf = ['molecular_function unknown', 'ATP binding', 'kinase activity', 'DNA binding']
        
        gene_go = pd.DataFrame({
            'description': [f'Description for gene {i}' for i in range(100)],
            'go_BioProc': [':'.join(np.random.choice(go_terms_bp, np.random.randint(1, 4))) for _ in range(100)],
            'go_CellComp': [':'.join(np.random.choice(go_terms_cc, np.random.randint(1, 3))) for _ in range(100)],
            'go_MolFunc': [':'.join(np.random.choice(go_terms_mf, np.random.randint(1, 2))) for _ in range(100)],
            'proteinname': [f'Protein{i}p' for i in range(100)]
        })
        
        # Dummy gene_locus
        gene_locus = pd.DataFrame({
            'stdname': [f'GENE{i}' for i in range(100)],
            'sysname': [f'YAL{i:03d}W' for i in range(100)]
        })
    
    # Menggabungkan dataset
    print("Menggabungkan dataset...")
    
    # Langkah 1: Gabungkan gene_locus dengan gene_seq berdasarkan sysname
    data_seq_locus = pd.merge(gene_seq, gene_locus, on='sysname', how='inner')
    
    # Langkah 2: Gabungkan hasil dengan gene_go berdasarkan proteinname dan stdname
    # Catatan: Asumsi bahwa proteinname di gene_go bisa dipetakan ke stdname di gene_locus
    # Jika tidak ada hubungan langsung, kita perlu mencari cara lain untuk menghubungkan data
    
    # Untuk contoh ini, kita asumsikan bahwa proteinname di gene_go memiliki format "Namep" 
    # dan stdname di gene_locus adalah "NAME"
    if 'proteinname' in gene_go.columns:
        # Ekstrak nama protein tanpa akhiran 'p'
        gene_go['protein_base'] = gene_go['proteinname'].str.replace('p$', '', regex=True)
        
        # Ubah ke uppercase untuk mencocokkan dengan stdname
        gene_go['protein_base'] = gene_go['protein_base'].str.upper()
        
        # Gabungkan berdasarkan protein_base dan stdname
        data = pd.merge(data_seq_locus, gene_go, left_on='stdname', right_on='protein_base', how='inner')
        
        # Hapus kolom sementara
        data = data.drop('protein_base', axis=1)
    else:
        # Jika tidak ada kolom proteinname, kita gunakan pendekatan lain
        # Misalnya, kita bisa mencoba mencocokkan berdasarkan sysname jika ada di gene_go
        print("Kolom proteinname tidak ditemukan di gene_go. Menggunakan pendekatan alternatif...")
        
        # Untuk demonstrasi, kita akan menggabungkan data secara acak
        # Dalam implementasi nyata, Anda perlu menemukan cara yang tepat untuk menghubungkan data
        data = data_seq_locus.iloc[:min(len(data_seq_locus), len(gene_go))].copy()
        for col in gene_go.columns:
            data[col] = gene_go[col].iloc[:len(data)].values
    
    print(f"Data setelah digabungkan: {len(data)} baris")
    
    # Ekstraksi fitur k-mer dari urutan DNA
    print("Mengekstraksi fitur k-mer dari urutan DNA...")
    X = []
    all_kmers = set()
    
    # Ekstraksi k-mer dari 100 data pertama untuk membuat kamus fitur
    for seq in data['gene'].iloc[:min(len(data), 100)]:
        if isinstance(seq, str):
            kmers = extract_kmer_features(seq)
            all_kmers.update(kmers.keys())
    
    all_kmers = list(all_kmers)
    print(f"Jumlah fitur k-mer unik: {len(all_kmers)}")
    
    # Membuat vektor fitur untuk semua data
    for seq in data['gene']:
        if isinstance(seq, str):
            kmers = extract_kmer_features(seq)
            features = [kmers.get(kmer, 0) for kmer in all_kmers]
        else:
            features = [0] * len(all_kmers)
        X.append(features)
    
    X = np.array(X)
    print(f"Dimensi fitur: {X.shape}")
    
    # Memproses label GO
    print("Memproses label Gene Ontology...")
    
    # Fungsi untuk memproses label GO
    def process_go_labels(go_column):
        if go_column.dtype == 'object':
            return go_column.apply(lambda x: x.split(':') if isinstance(x, str) else [])
        else:
            return go_column.apply(lambda x: [])
    
    # Memproses label untuk setiap kategori GO
    bp_labels = process_go_labels(data['go_BioProc'])
    cc_labels = process_go_labels(data['go_CellComp'])
    mf_labels = process_go_labels(data['go_MolFunc'])
    
    # Menggunakan MultiLabelBinarizer untuk mengubah label menjadi format one-hot
    mlb_bp = MultiLabelBinarizer()
    mlb_cc = MultiLabelBinarizer()
    mlb_mf = MultiLabelBinarizer()
    
    y_bp = mlb_bp.fit_transform(bp_labels)
    y_cc = mlb_cc.fit_transform(cc_labels)
    y_mf = mlb_mf.fit_transform(mf_labels)
    
    print(f"Jumlah label BP: {y_bp.shape[1]}")
    print(f"Jumlah label CC: {y_cc.shape[1]}")
    print(f"Jumlah label MF: {y_mf.shape[1]}")
    
    # Menggabungkan semua label
    y = np.hstack((y_bp, y_cc, y_mf))
    
    # Menyimpan informasi tentang label
    label_info = {
        'bp_classes': mlb_bp.classes_,
        'cc_classes': mlb_cc.classes_,
        'mf_classes': mlb_mf.classes_,
        'bp_count': y_bp.shape[1],
        'cc_count': y_cc.shape[1],
        'mf_count': y_mf.shape[1]
    }
    
    # Membagi data menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, label_info, data

# Fungsi untuk melatih model
def train_model(X_train, y_train):
    """
    Melatih model Random Forest untuk multi-label classification
    
    Args:
        X_train (np.array): Data fitur untuk pelatihan
        y_train (np.array): Data label untuk pelatihan
        
    Returns:
        MultiOutputClassifier: Model yang telah dilatih
    """
    print("Melatih model Random Forest...")
    
    # Menggunakan Random Forest sebagai base classifier
    base_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    # Menggunakan MultiOutputClassifier untuk multi-label classification
    model = MultiOutputClassifier(base_clf)
    
    # Melatih model
    model.fit(X_train, y_train)
    
    return model

# Fungsi untuk mengevaluasi model
def evaluate_model(model, X_test, y_test, label_info):
    """
    Mengevaluasi model menggunakan berbagai metrik
    
    Args:
        model (MultiOutputClassifier): Model yang telah dilatih
        X_test (np.array): Data fitur untuk pengujian
        y_test (np.array): Data label untuk pengujian
        label_info (dict): Informasi tentang label
        
    Returns:
        dict: Hasil evaluasi
    """
    print("Mengevaluasi model...")
    
    # Memprediksi label
    y_pred = model.predict(X_test)
    
    # Menghitung metrik evaluasi
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    macro_precision = precision_score(y_test, y_pred, average='macro')
    macro_recall = recall_score(y_test, y_pred, average='macro')
    h_loss = hamming_loss(y_test, y_pred)
    
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Hamming Loss: {h_loss:.4f}")
    
    # Menghitung metrik per kategori GO
    bp_count = label_info['bp_count']
    cc_count = label_info['cc_count']
    mf_count = label_info['mf_count']
    
    y_test_bp = y_test[:, :bp_count]
    y_test_cc = y_test[:, bp_count:bp_count+cc_count]
    y_test_mf = y_test[:, bp_count+cc_count:]
    
    y_pred_bp = y_pred[:, :bp_count]
    y_pred_cc = y_pred[:, bp_count:bp_count+cc_count]
    y_pred_mf = y_pred[:, bp_count+cc_count:]
    
    bp_f1 = f1_score(y_test_bp, y_pred_bp, average='macro')
    cc_f1 = f1_score(y_test_cc, y_pred_cc, average='macro')
    mf_f1 = f1_score(y_test_mf, y_pred_mf, average='macro')
    
    print(f"BP F1 Score: {bp_f1:.4f}")
    print(f"CC F1 Score: {cc_f1:.4f}")
    print(f"MF F1 Score: {mf_f1:.4f}")
    
    # Visualisasi hasil
    categories = ['BP', 'CC', 'MF', 'Overall']
    f1_scores = [bp_f1, cc_f1, mf_f1, macro_f1]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=categories, y=f1_scores)
    plt.title('F1 Score per Kategori Gene Ontology')
    plt.ylabel('F1 Score')
    plt.ylim(0, 1)
    plt.savefig('f1_scores.png')
    plt.close()
    
    # Mengembalikan hasil evaluasi
    return {
        'macro_f1': macro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'hamming_loss': h_loss,
        'bp_f1': bp_f1,
        'cc_f1': cc_f1,
        'mf_f1': mf_f1
    }

# Fungsi untuk menyimpan model
def save_model(model, label_info, filename='dna_annotation_model.pkl'):
    """
    Menyimpan model dan informasi label
    
    Args:
        model (MultiOutputClassifier): Model yang telah dilatih
        label_info (dict): Informasi tentang label
        filename (str): Nama file untuk menyimpan model
    """
    print(f"Menyimpan model ke {filename}...")
    
    # Menyimpan model dan informasi label
    with open(filename, 'wb') as f:
        pickle.dump({'model': model, 'label_info': label_info}, f)

# Fungsi untuk memuat model
def load_model(filename='dna_annotation_model.pkl'):
    """
    Memuat model dan informasi label
    
    Args:
        filename (str): Nama file model
        
    Returns:
        tuple: model, label_info
    """
    print(f"Memuat model dari {filename}...")
    
    # Memuat model dan informasi label
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    return data['model'], data['label_info']

# Fungsi untuk memprediksi anotasi GO untuk urutan DNA baru
def predict_go_terms(sequence, model, label_info, k=3):
    """
    Memprediksi anotasi GO untuk urutan DNA baru
    
    Args:
        sequence (str): Urutan DNA
        model (MultiOutputClassifier): Model yang telah dilatih
        label_info (dict): Informasi tentang label
        k (int): Panjang k-mer
        
    Returns:
        dict: Hasil prediksi
    """
    print("Memprediksi anotasi GO untuk urutan DNA baru...")
    
    # Ekstraksi fitur k-mer
    kmers = extract_kmer_features(sequence, k)
    
    # Membuat vektor fitur
    all_kmers = set()
    for i in range(len(sequence) - k + 1):
        all_kmers.add(sequence[i:i+k])
    
    features = [kmers.get(kmer, 0) for kmer in all_kmers]
    
    # Memprediksi label
    y_pred = model.predict([features])[0]
    
    # Memisahkan hasil prediksi berdasarkan kategori GO
    bp_count = label_info['bp_count']
    cc_count = label_info['cc_count']
    mf_count = label_info['mf_count']
    
    y_pred_bp = y_pred[:bp_count]
    y_pred_cc = y_pred[bp_count:bp_count+cc_count]
    y_pred_mf = y_pred[bp_count+cc_count:]
    
    # Mendapatkan label GO yang diprediksi
    bp_classes = label_info['bp_classes']
    cc_classes = label_info['cc_classes']
    mf_classes = label_info['mf_classes']
    
    predicted_bp = [bp_classes[i] for i in range(len(y_pred_bp)) if y_pred_bp[i] == 1]
    predicted_cc = [cc_classes[i] for i in range(len(y_pred_cc)) if y_pred_cc[i] == 1]
    predicted_mf = [mf_classes[i] for i in range(len(y_pred_mf)) if y_pred_mf[i] == 1]
    
    # Mengembalikan hasil prediksi
    return {
        'BP': predicted_bp,
        'CC': predicted_cc,
        'MF': predicted_mf
    }

# Fungsi utama
def main():
    """
    Fungsi utama untuk melatih dan mengevaluasi model
    """
    # Memuat dan memproses data
    X_train, X_test, y_train, y_test, label_info, data = load_and_process_data()
    
    # Melatih model
    model = train_model(X_train, y_train)
    
    # Mengevaluasi model
    evaluate_model(model, X_test, y_test, label_info)
    
    # Menyimpan model
    save_model(model, label_info)
    
    # Contoh prediksi untuk urutan DNA baru
    sample_sequence = data['gene'].iloc[0]
    print("\nContoh prediksi untuk urutan DNA:")
    print(sample_sequence[:50] + "...")
    
    predictions = predict_go_terms(sample_sequence, model, label_info)
    
    print("\nHasil prediksi:")
    print("Biological Process (BP):", predictions['BP'])
    print("Cellular Component (CC):", predictions['CC'])
    print("Molecular Function (MF):", predictions['MF'])

if __name__ == "__main__":
    main()