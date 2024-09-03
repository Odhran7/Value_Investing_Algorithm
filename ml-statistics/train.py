import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

scaler = StandardScaler()
df = pd.read_csv('./merged_encoded_data_2020_election.csv')
df = df.drop(columns=['state.1', 'name', 'fips', 'majority_Trump'], axis=1, inplace=False)
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit_transform(df)
df_imputed = pd.DataFrame(imp.fit_transform(df), columns=df.columns)
X = df_imputed.drop('majority_Biden', axis=1, inplace=False)
X = pd.DataFrame(scaler.fit_transform(X))
y = df_imputed['majority_Biden']

# First I want to oversample the minority class
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.regularizers import l2
from sklearn.metrics import make_scorer, f1_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD

early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

def create_model(dropout_rate=0.0, regularization_rate=0.0, optimizer='adam', init_mode='uniform', activation='relu', neurons=161, layers=3, learning_rate=0.001):
    model = Sequential()
    model.add(Dense(neurons, input_dim=len(X_train_resampled.columns), activation=activation, kernel_initializer=init_mode, kernel_regularizer=l2(regularization_rate)))
    model.add(Dropout(dropout_rate))
    for i in range(layers - 1):
        model.add(Dense(neurons // (2 ** (i + 1)), activation=activation, kernel_initializer=init_mode, kernel_regularizer=l2(regularization_rate)))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = SGD(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

model = KerasClassifier(model=create_model, epochs=100, batch_size=64, verbose=1, callbacks=[early_stopping])

param_grid = {
    'model__dropout_rate': [0.2, 0.3, 0.4, 0.5],
    'model__regularization_rate': [0.001, 0.01, 0.05],
    'model__optimizer': ['adam', 'sgd'],
    'model__init_mode': ['uniform', 'normal', 'he_normal'],
    'model__activation': ['relu', 'tanh', 'sigmoid'],
    'model__neurons': [50, 100, 150],
    'model__layers': [2, 3, 4],
    'batch_size': [32, 64, 128],
    'epochs': [50, 100, 150],
    'model__learning_rate': [0.001, 0.01, 0.1]
}

f1_scorer = make_scorer(f1_score)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=f1_scorer, n_jobs=2, cv=3)
grid_result = grid.fit(X_train_resampled, y_train_resampled)
print(f'Best F1 Score: {grid_result.best_score_} using {grid_result.best_params_}')
