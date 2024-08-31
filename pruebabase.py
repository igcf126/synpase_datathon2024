import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load all the datasets
consumos_df = pd.read_csv('./data/df_entrenamiento/consumos_participantes.csv')
cobros_df = pd.read_csv('./data/df_entrenamiento/df_cobros_participantes.csv')
loans_df = pd.read_csv('./data/df_entrenamiento/df_loans_participantes.csv')
quejas_df = pd.read_csv('./data/df_entrenamiento/df_quejas_participantes.csv')
target_df = pd.read_csv('./data/df_entrenamiento/df_target_participantes.csv')

# Merge the dataframes
df_merged = target_df.merge(consumos_df, on='id', how='left') \
                     .merge(cobros_df, on='id', how='left') \
                     .merge(loans_df, on='id', how='left') \
                     .merge(quejas_df, on='id', how='left')

# Drop 'id' if not needed for modeling
df_merged = df_merged.drop(columns=['id'])

# Assuming the target column is named 'churn'
X = df_merged.drop(columns=['churn'])
y = df_merged['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

# Classification Report
class_report = classification_report(y_test, y_pred)
print('Classification Report:')
print(class_report)

from xgboost import plot_importance
import matplotlib.pyplot as plt

# Plot feature importance
plot_importance(model)
plt.show()
