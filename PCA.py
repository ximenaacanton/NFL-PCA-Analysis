import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter

# Load NFL teams data
df_teams = pd.read_csv("nfl_teams.csv")
df_teams_cleaned = df_teams.dropna(subset=['team_division'])

# Load stadiums data
df_stadiums = pd.read_csv('nfl_stadiums.csv', encoding='latin1')

# Map stadium open/closed status
stadium_open_status = {
     'Acrisure Stadium': 'Open',
    'Alamo Dome': 'Closed',
    'Allegiant Stadium': 'Closed',
    'Allianz Arena': 'Closed',
    'Alumni Stadium': 'Open',
    'Anaheim Stadium': 'Open',
    'Arrowhead Stadium': 'Open',
    'AT&T Stadium': 'Closed',
    'Atlanta-Fulton County Stadium': 'Open',
    'Balboa Stadium': 'Open',
    'Bank of America Stadium': 'Open',
    'Bills Stadium': 'Open',
    'Busch Memorial Stadium': 'Open',
    'Caesars Superdome': 'Closed',
    'Candlestick Park': 'Open',
    'CenturyLink Field': 'Closed',
    'Cinergy Field': 'Open',
    'Cleveland Municipal Stadium': 'Open',
    'Cotton Bowl': 'Open',
    'Cowboys Stadium': 'Closed',
    'Dignity Health Sports Park': 'Open',
    'Edward Jones Dome': 'Closed',
    'Empower Field at Mile High': 'Open',
    'Estadio Azteca': 'Open',
    'EverBank Field': 'Open',
    'FedEx Field': 'Open',
    'FirstEnergy Stadium': 'Open',
    'Ford Field': 'Closed',
    'Foxboro Stadium': 'Open',
    'Frankfurt Stadium': 'Open',
    'Franklin Field': 'Open',
    'GEHA Field at Arrowhead Stadium': 'Open',
    'Georgia Dome': 'Closed',
    'Giants Stadium': 'Closed',
    'Gillette Stadium': 'Open',
    'Hard Rock Stadium': 'Open',
    'Harvard Stadium': 'Open',
    'Heinz Field': 'Open',
    'Highmark Stadium': 'Open',
    "Houlihan's Stadium": 'Open',
    'Houston Astrodome': 'Closed',
    'Hubert H. Humphrey Metrodome': 'Closed',
    'Husky Stadium': 'Open',
    'Kansas City Municipal Stadium': 'Open',
    'Kezar Stadium': 'Open',
    'Lambeau Field': 'Open',
    "Levi's Stadium": 'Open',
    'Liberty Bowl Memorial Stadium': 'Open',
    'Lincoln Financial Field': 'Open',
    'Los Angeles Memorial Coliseum': 'Open',
    'Louisiana Superdome': 'Closed',
    'LP Stadium': 'Open',
    'Lucas Oil Stadium': 'Closed',
    'Lumen Field': 'Closed',
    'M&T Bank Stadium': 'Open',
    'Mall of America Field': 'Closed',
    'Memorial Stadium (Baltimore)': 'Open',
    'Memorial Stadium (Champaign)': 'Open',
    'Memorial Stadium (Clemson)': 'Open',
    'Mercedes-Benz Stadium': 'Closed',
    'Mercedes-Benz Superdome': 'Closed',
    'MetLife Stadium': 'Open',
    'Metropolitan Stadium': 'Open',
    'Mile High Stadium': 'Open',
    'New Era Field': 'Open',
    'Nippert Stadium': 'Open',
    'Nissan Stadium': 'Open',
    'NRG Stadium': 'Closed',
    'Oakland Coliseum': 'Open',
    'Orange Bowl': 'Open',
    'Paul Brown Stadium': 'Open',
    'Paycor Stadium': 'Open',
    'Pitt Stadium': 'Open',
    'Pontiac Silverdome': 'Closed',
    'Qualcomm Stadium': 'Open',
    'Ralph Wilson Stadium': 'Open',
    'Raymond James Stadium': 'Open',
    'RCA Dome': 'Closed',
    'Reliant Stadium': 'Closed',
    'Alltel Stadium': 'Open',
    'Dolphin Stadium': 'Open',
    'Fenway Park': 'Open',
    'Jack Murphy Stadium': 'Open',
    'Joe Robbie Stadium': 'Open',
    'Legion Field': 'Open',
    'Pro Player Stadium': 'Open',
    'RFK Memorial Stadium': 'Open',
    'Rice Stadium': 'Open',
    'Rogers Centre': 'Closed',
    'Rose Bowl': 'Open',
    'Seattle Kingdome': 'Closed',
    'Shea Stadium': 'Open',
    'SoFi Stadium': 'Closed',
    'Soldier Field': 'Open',
    'Sports Authority Field at Mile High': 'Open',
    'Stanford Stadium': 'Open',
    'State Farm Stadium': 'Closed',
    'StubHub Center': 'Open',
    'Sun Devil Stadium': 'Open',
    'Sun Life Stadium': 'Open',
    'Tampa Stadium': 'Open',
    'TCF Bank Stadium': 'Open',
    'Texas Stadium': 'Closed',
    'Three Rivers Stadium': 'Closed',
    'TIAA Bank Field': 'Open',
    'Tiger Stadium': 'Open',
    'Tiger Stadium (LSU)': 'Open',
    'Tottenham Hotspur Stadium': 'Closed',
    'Tottenham Stadium': 'Closed',
    'Tulane Stadium': 'Open',
    'Twickenham Stadium': 'Open',
    'U.S. Bank Stadium': 'Closed',
    'University of Phoenix Stadium': 'Closed',
    'Vanderbilt Stadium': 'Open',
    'Veterans Stadium': 'Closed',
    'War Memorial Stadium': 'Open',
    'Wembley Stadium': 'Closed',
    'Wrigley Field': 'Open',
    'Yale Bowl': 'Open',
    'Yankee Stadium': 'Open',
}
df_stadiums['stadium_type'] = df_stadiums['stadium_name'].map(stadium_open_status)

# Load scores data
df_scores = pd.read_csv('spreadspoke_scores.csv')
df_scores_baltimore = df_scores[(df_scores['team_home'] == 'Baltimore Ravens') | (df_scores['team_away'] == 'Baltimore Ravens')]

# Drop unnecessary columns
df_scores_baltimore.drop(columns=['Unnamed: 17', 'Unnamed: 18', 'Unnamed: 19', 'Unnamed: 20', 'Unnamed: 21', 'Unnamed: 22', 'Unnamed: 23', 'Unnamed: 24'], inplace=True)

# Feature engineering for modeling
df_features = df_scores_baltimore.drop(['schedule_date', 'stadium', 'weather_detail'], axis=1)
le = LabelEncoder()
df_features['team_home'] = le.fit_transform(df_features['team_home'])
df_features['team_away'] = le.fit_transform(df_features['team_away'])
df_features['team_favorite_id'] = le.fit_transform(df_features['team_favorite_id'])

sns.pairplot(df_features[['score_home', 'score_away', 'spread_favorite', 'over_under_line', 'weather_temperature', 'weather_wind_mph', 'weather_humidity']])


df_features_numeric = df_features.select_dtypes(include=['number'])
# Supongamos que tu DataFrame se llama df_features
df_features.head()
plt.figure(figsize=(10, 8))
corr_matrix = df_features_numeric.corr()  # Calcula la matriz de correlación
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)  # Muestra los coeficientes de correlación
plt.show()

# Create target variable
# Adjust for home/away games
df_features['win_loss'] = (df_features['score_home'] > df_features['score_away']).astype(int)
df_features.loc[df_scores_baltimore['team_away'] == 'Baltimore Ravens', 'win_loss'] = (
    df_features['score_away'] > df_features['score_home']
).astype(int)

# Prepare features (X) and target (y)
X = df_features.drop(['score_home', 'score_away', 'win_loss', 'schedule_week'], axis=1)
y = df_features['win_loss']

# Fill missing values and scale features
X['spread_favorite'] = X['spread_favorite'].fillna(X['spread_favorite'].mean())
X['over_under_line'] = X['over_under_line'].fillna(0).astype(float)
X.loc[X['stadium_neutral'] == True, ['weather_temperature', 'weather_wind_mph', 'weather_humidity']] = 0
X['weather_temperature'] = X['weather_temperature'].fillna(X['weather_temperature'].mean())
X['weather_wind_mph'] = X['weather_wind_mph'].fillna(X['weather_wind_mph'].mean())
X['weather_humidity'] = X['weather_humidity'].fillna(X['weather_humidity'].mean())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
target_variance = 0.95
pca = PCA(n_components=target_variance)
X_pca = pca.fit_transform(X_scaled)

# Este paso muestra la cantidad de varianza explicada por cada componente principal.
# Ayuda a entender cuántos componentes contribuyen significativamente a la variabilidad de los datos.
print("Explained Variance Ratio:", pca.explained_variance_ratio_)
print("Number of Components:", pca.n_components_)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Train RandomForest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("RandomForest Classifier Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Train Logistic Regression model
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))
print("Classification Report:")
print(classification_report(y_test, y_pred_lr))

# GridSearch for Logistic Regression
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'lbfgs', 'saga'],
    'penalty': ['l2']
}
grid_search = GridSearchCV(LogisticRegression(random_state=42, max_iter=1000), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Model Accuracy: {accuracy_score(y_test, y_pred_best):.2f}")

# Explained variance by component
explained_variance = pca.explained_variance_ratio_
plt.figure(figsize=(10, 6))
# Este gráfico de barras muestra la varianza explicada por cada componente principal,
# útil para decidir cuántos componentes retener en el análisis.
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, align='center')
plt.xlabel('Principal Component Index')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance by Principal Component')
plt.show()

# Scree Plot
plt.figure(figsize=(10, 6))
# El gráfico de Scree ayuda a identificar el número óptimo de componentes principales.
# Generalmente, se busca un 'codo' en el gráfico donde la varianza explicada comienza a disminuir más lentamente.
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
plt.xlabel('Principal Component Index')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')
plt.show()

# Heatmap of Component Loadings
loadings = pca.components_.T
feature_names = X.columns
loadings_df = pd.DataFrame(loadings, columns=[f'PC{i+1}' for i in range(pca.n_components_)], index=feature_names)
plt.figure(figsize=(10, 8))
# El mapa de calor de las cargas muestra la relación entre las variables originales y los componentes principales,
# facilitando la interpretación de qué variables tienen más peso en cada componente.
sns.heatmap(loadings_df, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Heatmap of Principal Component Loadings')
plt.xlabel('Principal Components')
plt.ylabel('Features')
plt.show()

# Biplot
plt.figure(figsize=(12, 8))
# El biplot permite visualizar tanto la proyección de los datos en los componentes principales
# como la dirección de las variables originales. Esto ayuda a interpretar la estructura de los datos en el espacio reducido.
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.5)
for i, feature in enumerate(feature_names):
    plt.arrow(0, 0, loadings[i, 0] * max(X_pca[:, 0]), loadings[i, 1] * max(X_pca[:, 1]),
              color='r', alpha=0.7, head_width=0.05)
    plt.text(loadings[i, 0] * max(X_pca[:, 0]) * 1.1, loadings[i, 1] * max(X_pca[:, 1]) * 1.1,
             feature, color='r', ha='center', va='center')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Biplot of Principal Components and Feature Loadings')
plt.grid()
plt.show()

# Projections of Data
plt.figure(figsize=(10, 6))
# La proyección de los datos sobre los primeros dos componentes principales permite identificar
# patrones o clusters que no eran evidentes en el espacio original de las variables.
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Projections of Data on Principal Components')
plt.colorbar(label='Win/Loss')
plt.grid()
plt.show()

# Train-test split without PCA
X_train_no_pca, X_test_no_pca, y_train_no_pca, y_test_no_pca = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train RandomForest model without PCA
rf_model_no_pca = RandomForestClassifier(random_state=42)
rf_model_no_pca.fit(X_train_no_pca, y_train_no_pca)
y_pred_rf_no_pca = rf_model_no_pca.predict(X_test_no_pca)
print("RandomForest Classifier without PCA Accuracy:", accuracy_score(y_test_no_pca, y_pred_rf_no_pca))
print("Classification Report without PCA for Random Forest:")
print(classification_report(y_test_no_pca, y_pred_rf_no_pca))

# Train Logistic Regression model without PCA
lr_model_no_pca = LogisticRegression(random_state=42, max_iter=1000)
lr_model_no_pca.fit(X_train_no_pca, y_train_no_pca)
y_pred_lr_no_pca = lr_model_no_pca.predict(X_test_no_pca)
print("Logistic Regression without PCA Accuracy:", accuracy_score(y_test_no_pca, y_pred_lr_no_pca))
print("Confusion Matrix without PCA for Logistic Regression:")
print(confusion_matrix(y_test_no_pca, y_pred_lr_no_pca))
print("Classification Report without PCA for Logistic Regression:")
print(classification_report(y_test_no_pca, y_pred_lr_no_pca))

# Comparison Summary
print("\nComparison Summary:\n")
print(f"RandomForest Accuracy with PCA: {accuracy_score(y_test, y_pred_rf):.2f}")
print(f"RandomForest Accuracy without PCA: {accuracy_score(y_test_no_pca, y_pred_rf_no_pca):.2f}")
print(f"Logistic Regression Accuracy with PCA: {accuracy_score(y_test, y_pred_lr):.2f}")
print(f"Logistic Regression Accuracy without PCA: {accuracy_score(y_test_no_pca, y_pred_lr_no_pca):.2f}")

# Prediction for next two games of Baltimore Ravens
# Creating data for the next two games
df_next_games = pd.DataFrame({
    'team_home': ['Baltimore Ravens', 'New York Giants'],
    'team_away': ['Philadelphia Eagles', 'Baltimore Ravens'],
    'team_favorite_id': ['Baltimore Ravens', 'Baltimore Ravens'],
    'spread_favorite': [2.5, -1.5],
    'over_under_line': [47.5, 44.0],
    'stadium_neutral': [False, False],
    'weather_temperature': [70, 68],
    'weather_wind_mph': [12, 9],
    'weather_humidity': [65, 60],
    'schedule_playoff': [0, 0],
    'schedule_season': [2024, 2024]
})

# Reorder columns to match the order during model training
df_next_games = df_next_games[X.columns]

# Ensure all team names are seen by the LabelEncoder
all_teams = df_scores_baltimore['team_home'].unique().tolist() + df_scores_baltimore['team_away'].unique().tolist()
le.fit(all_teams)

# Preprocessing the new data
df_next_games['team_home'] = le.transform(df_next_games['team_home'])
df_next_games['team_away'] = le.transform(df_next_games['team_away'])
df_next_games['team_favorite_id'] = le.transform(df_next_games['team_favorite_id'])

# Scaling the features
X_next_games_scaled = scaler.transform(df_next_games)

# Applying PCA to the new data
X_next_games_pca = pca.transform(X_next_games_scaled)

# Predicting the outcomes
predictions_next_games = rf_model.predict(X_next_games_pca)
print("Predictions for the next two games of Baltimore Ravens:", predictions_next_games)

# Visualización de varianza acumulada
explained_variance_cumsum = np.cumsum(pca.explained_variance_ratio_)
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance_cumsum) + 1), explained_variance_cumsum, marker='o', linestyle='--')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance by Principal Component')
plt.axhline(y=0.95, color='r', linestyle='--')
plt.text(len(explained_variance_cumsum) * 0.9, 0.93, '95% Threshold', color='red')
plt.grid()
plt.show()

