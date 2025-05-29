import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv

# Fungsi untuk membangun graf Gene Ontology
def build_go_graph(go_terms, relationships):
    """
    Membangun graf Gene Ontology
    
    Args:
        go_terms (list): Daftar term GO
        relationships (list): Daftar hubungan antar term GO (parent-child)
        
    Returns:
        nx.DiGraph: Graf GO
    """
    print("Membangun graf Gene Ontology...")
    
    # Membuat graf berarah
    G = nx.DiGraph()
    
    # Menambahkan node
    for term in go_terms:
        G.add_node(term)
    
    # Menambahkan edge
    for parent, child in relationships:
        G.add_edge(parent, child)
    
    print(f"Graf GO berhasil dibangun dengan {G.number_of_nodes()} node dan {G.number_of_edges()} edge")
    
    return G

# Fungsi untuk memvisualisasikan graf GO
def visualize_go_graph(G, output_file='go_graph.png'):
    """
    Memvisualisasikan graf Gene Ontology
    
    Args:
        G (nx.DiGraph): Graf GO
        output_file (str): Nama file output
    """
    print("Memvisualisasikan graf Gene Ontology...")
    
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, 
            font_size=8, font_weight='bold', arrows=True)
    plt.savefig(output_file)
    plt.close()

# Kelas untuk model Graph Neural Network
class GNNModel(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GNNModel, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, h_feats)
        self.fc = nn.Linear(h_feats, num_classes)
        
    def forward(self, g, features):
        x = F.relu(self.conv1(g, features))
        x = F.relu(self.conv2(g, x))
        x = self.fc(x)
        return x

# Fungsi untuk melatih model GNN
def train_gnn(G, features, labels, epochs=100):
    """
    Melatih model Graph Neural Network
    
    Args:
        G (nx.DiGraph): Graf GO
        features (torch.Tensor): Fitur node
        labels (torch.Tensor): Label node
        epochs (int): Jumlah epoch
        
    Returns:
        GNNModel: Model yang telah dilatih
    """
    print("Melatih model Graph Neural Network...")
    
    # Mengubah graf NetworkX menjadi graf DGL
    dgl_G = dgl.from_networkx(G)
    
    # Membagi data menjadi data latih dan data uji
    idx = torch.arange(features.shape[0])
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)
    
    # Inisialisasi model
    in_feats = features.shape[1]
    h_feats = 64
    num_classes = labels.shape[1]
    model = GNNModel(in_feats, h_feats, num_classes)
    
    # Optimizer dan loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCEWithLogitsLoss()
    
    # Melatih model
    for epoch in range(epochs):
        model.train()
        
        # Forward pass
        logits = model(dgl_G, features)
        loss = loss_fn(logits[train_idx], labels[train_idx])
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    # Evaluasi model
    model.eval()
    with torch.no_grad():
        logits = model(dgl_G, features)
        pred = (torch.sigmoid(logits[test_idx]) > 0.5).float()
        
        # Menghitung metrik evaluasi
        f1 = f1_score(labels[test_idx].numpy(), pred.numpy(), average='macro')
        precision = precision_score(labels[test_idx].numpy(), pred.numpy(), average='macro')
        recall = recall_score(labels[test_idx].numpy(), pred.numpy(), average='macro')
        
        print(f"F1 Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
    
    return model

# Fungsi untuk memperbaiki prediksi berdasarkan struktur GO
def refine_predictions(predictions, G):
    """
    Memperbaiki prediksi berdasarkan struktur hierarki GO
    
    Args:
        predictions (dict): Hasil prediksi dari model dasar
        G (nx.DiGraph): Graf GO
        
    Returns:
        dict: Hasil prediksi yang telah diperbaiki
    """
    print("Memperbaiki prediksi berdasarkan struktur hierarki GO...")
    
    refined_predictions = {
        'BP': set(predictions['BP']),
        'CC': set(predictions['CC']),
        'MF': set(predictions['MF'])
    }
    
    # Fungsi untuk menambahkan semua parent term
    def add_parents(term, category):
        if term in G:
            for parent in G.predecessors(term):
                refined_predictions[category].add(parent)
                add_parents(parent, category)
    
    # Menambahkan parent term untuk setiap prediksi
    for term in predictions['BP']:
        add_parents(term, 'BP')
    
    for term in predictions['CC']:
        add_parents(term, 'CC')
    
    for term in predictions['MF']:
        add_parents(term, 'MF')
    
    # Mengubah set menjadi list
    refined_predictions = {
        'BP': list(refined_predictions['BP']),
        'CC': list(refined_predictions['CC']),
        'MF': list(refined_predictions['MF'])
    }
    
    return refined_predictions

# Fungsi untuk memuat data GO dan membangun graf
def load_go_data():
    """
    Memuat data GO dan membangun graf
    
    Returns:
        tuple: go_terms, relationships, G
    """
    print("Memuat data Gene Ontology...")
    
    try:
        # Memuat data GO dari file
        # Catatan: Dalam implementasi nyata, Anda perlu memuat data GO dari sumber yang sesuai
        go_data = pd.read_csv('go_data.csv')
        
        go_terms = go_data['term'].tolist()
        relationships = list(zip(go_data['parent'], go_data['child']))
        
    except FileNotFoundError:
        print("File data GO tidak ditemukan. Menggunakan data dummy...")
        
        # Membuat data dummy
        go_terms = [
            'GO:0008150', 'GO:0009987', 'GO:0008152', 'GO:0071704', 'GO:0044237',
            'GO:0044238', 'GO:0005975', 'GO:0006006', 'GO:0019318', 'GO:0005996',
            'GO:0006091', 'GO:0006007', 'GO:0006096', 'GO:0061621', 'GO:0006094',
            'GO:0006099', 'GO:0005576', 'GO:0005623', 'GO:0005622', 'GO:0005737',
            'GO:0044444', 'GO:0044424', 'GO:0043226', 'GO:0043229', 'GO:0043227',
            'GO:0043231', 'GO:0005739', 'GO:0005740', 'GO:0003674', 'GO:0003824',
            'GO:0016787', 'GO:0016798', 'GO:0004553', 'GO:0016832', 'GO:0004034'
        ]
        
        # Membuat hubungan hierarki
        relationships = [
            ('GO:0008150', 'GO:0009987'), ('GO:0008150', 'GO:0008152'),
            ('GO:0009987', 'GO:0071704'), ('GO:0008152', 'GO:0071704'),
            ('GO:0071704', 'GO:0044237'), ('GO:0071704', 'GO:0044238'),
            ('GO:0044238', 'GO:0005975'), ('GO:0005975', 'GO:0006006'),
            ('GO:0006006', 'GO:0019318'), ('GO:0019318', 'GO:0005996'),
            ('GO:0005996', 'GO:0006091'), ('GO:0006091', 'GO:0006007'),
            ('GO:0006007', 'GO:0006096'), ('GO:0006096', 'GO:0061621'),
            ('GO:0061621', 'GO:0006094'), ('GO:0006094', 'GO:0006099'),
            ('GO:0005576', 'GO:0005623'), ('GO:0005623', 'GO:0005622'),
            ('GO:0005622', 'GO:0005737'), ('GO:0005737', 'GO:0044444'),
            ('GO:0044444', 'GO:0044424'), ('GO:0044424', 'GO:0043226'),
            ('GO:0043226', 'GO:0043229'), ('GO:0043229', 'GO:0043227'),
            ('GO:0043227', 'GO:0043231'), ('GO:0043231', 'GO:0005739'),
            ('GO:0005739', 'GO:0005740'), ('GO:0003674', 'GO:0003824'),
            ('GO:0003824', 'GO:0016787'), ('GO:0016787', 'GO:0016798'),
            ('GO:0016798', 'GO:0004553'), ('GO:0004553', 'GO:0016832'),
            ('GO:0016832', 'GO:0004034')
        ]
    
    # Membangun graf GO
    G = build_go_graph(go_terms, relationships)
    
    return go_terms, relationships, G

# Fungsi utama
def main():
    """
    Fungsi utama untuk membangun dan menguji model graph-based
    """
    # Memuat data GO dan membangun graf
    go_terms, relationships, G = load_go_data()
    
    # Memvisualisasikan graf GO
    visualize_go_graph(G)
    
    # Membuat fitur dummy untuk node
    features = torch.randn(len(go_terms), 10)
    
    # Membuat label dummy untuk node
    labels = torch.zeros(len(go_terms), 3)
    for i in range(len(go_terms)):
        labels[i, np.random.randint(0, 3)] = 1
    
    # Melatih model GNN
    model = train_gnn(G, features, labels)
    
    # Contoh penggunaan untuk memperbaiki prediksi
    predictions = {
        'BP': ['GO:0006096', 'GO:0006094'],
        'CC': ['GO:0005739'],
        'MF': ['GO:0016832']
    }
    
    refined_predictions = refine_predictions(predictions, G)
    
    print("\nPrediksi awal:")
    print("BP:", predictions['BP'])
    print("CC:", predictions['CC'])
    print("MF:", predictions['MF'])
    
    print("\nPrediksi yang diperbaiki:")
    print("BP:", refined_predictions['BP'])
    print("CC:", refined_predictions['CC'])
    print("MF:", refined_predictions['MF'])

if __name__ == "__main__":
    main()