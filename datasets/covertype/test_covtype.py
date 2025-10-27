import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, recall_score
from imblearn.metrics import geometric_mean_score
import xgboost as xgb
import time
import warnings

# Ignorar avisos
warnings.filterwarnings('ignore')

# Carregar os dados
print("Carregando dados...")
data = pd.read_csv('covtype.data', header=None)

# Separar features e labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Normalizar labels (1-7 para 0-6)
le = LabelEncoder()
y = le.fit_transform(y)

# Dividir em treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalizar features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Dicionário para armazenar resultados
results = {
    'Model': [],
    'Accuracy': [],
    'F1-Score': [],
    'G-Mean': [],
    'Training Time (s)': []
}

# Função para treinar e avaliar modelos
def train_and_evaluate(model, model_name):
    print(f"\nTreinando {model_name}...")
    start_time = time.time()
    
    # Treinar modelo
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Fazer previsões
    y_pred = model.predict(X_test)
    
    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    gmean = geometric_mean_score(y_test, y_pred, average='macro')
    
    # Armazenar resultados
    results['Model'].append(model_name)
    results['Accuracy'].append(accuracy)
    results['F1-Score'].append(f1)
    results['G-Mean'].append(gmean)
    results['Training Time (s)'].append(training_time)
    
    print(f"{model_name} treinado em {training_time:.2f} segundos")
    print(f"Acurácia: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"G-Mean: {gmean:.4f}")

# Inicializar modelos
models = {
    'MLP': MLPClassifier(
        hidden_layer_sizes=(100, 50),
        max_iter=300,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=10
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),
    'SVM': SVC(
        kernel='rbf',
        gamma='scale',
        random_state=42,
        max_iter=1000
    ),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=100,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
}

# Treinar e avaliar cada modelo
for name, model in models.items():
    train_and_evaluate(model, name)

# Exibir resultados finais
print("\n" + "="*60)
print("RESULTADOS FINAIS")
print("="*60)
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))