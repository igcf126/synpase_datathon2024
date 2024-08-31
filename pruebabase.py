import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Cargar los datasets
df_consumos = pd.read_csv("./data/df_entrenamiento/consumos_participantes.csv")
df_target = pd.read_csv("./data/df_entrenamiento/df_target_participantes.csv")
df_cobros = pd.read_csv("./data/df_entrenamiento/df_cobros_participantes.csv")
df_loans = pd.read_csv("./data/df_entrenamiento/df_loans_participantes.csv")
df_quejas = pd.read_csv("./data/df_entrenamiento/df_quejas_participantes.csv")

# Unir todas las tablas en una sola usando el ID como clave
df = df_target.merge(df_consumos, on='id', how='left') \
              .merge(df_cobros, on='id', how='left') \
              .merge(df_loans, on='id', how='left') \
              .merge(df_quejas, on='id', how='left')

# Eliminar la columna 'id' ya que no se usará para el modelo
df = df.drop(columns=['id'])

# Convertir fechas a formato de mes, día, año
date_columns = ['fecha_transaccion', 'fecha_activacion', 'fecha', 'fecha_apertura_reclamacion', 'fecha_resolucion_reclamacion']
for column in date_columns:
    if column in df.columns:
        df[column + '_mes'] = pd.to_datetime(df[column], errors='coerce').dt.month
        df[column + '_dia'] = pd.to_datetime(df[column], errors='coerce').dt.day
        df[column + '_anio'] = pd.to_datetime(df[column], errors='coerce').dt.year
        df = df.drop(columns=[column])

# Convertir las variables categóricas en numéricas usando LabelEncoder
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column].astype(str))
    label_encoders[column] = le

# Aplicar StandardScaler a las variables numéricas
scaler = StandardScaler()
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Separar en variables predictoras y objetivo (target)
X = df.drop(columns=['churned'])
y = df['churned']

# Dividir los datos en conjuntos de entrenamiento y prueba (75% entrenamiento, 25% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Usar LazyClassifier para entrenar varios modelos y seleccionar los mejores
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

# Imprimir el rendimiento de los modelos
print(models.head(5))

# Elegir los 5 mejores modelos para realizar GridSearchCV
top_models = models.index[:5]

# Diccionarios para almacenar los mejores parámetros y modelos
best_models = {}
best_params = {}

# Definir la búsqueda de hiperparámetros para cada modelo (esto es un ejemplo, ajusta según sea necesario)
param_grids = {
    'XGBClassifier': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    'RandomForestClassifier': {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20]
    },
    'LogisticRegression': {
        'C': [0.1, 1.0, 10.0],
        'solver': ['liblinear']
    },
    'KNeighborsClassifier': {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance']
    },
    'SVC': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    }
}

# GridSearch para cada modelo
for model_name in top_models:
    model_class = eval(model_name)
    grid_search = GridSearchCV(estimator=model_class(), param_grid=param_grids[model_name], cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_models[model_name] = grid_search.best_estimator()
    best_params[model_name] = grid_search.best_params()

# Imprimir los mejores parámetros y modelos
for model_name, model in best_models.items():
    print(f"Mejor modelo para {model_name}:")
    print(model)
    print(f"Mejores parámetros: {best_params[model_name]}")
    print("\n")