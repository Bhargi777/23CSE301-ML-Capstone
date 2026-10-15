"""Adult / Census Income preprocessing shared by the Review 2 classification notebooks.

Reproduces the Review 1 Part A pipeline exactly (same cleaning, log1p, engineered features,
stratified 80:20 split with random_state=42, one-hot encoder + StandardScaler fitted on the
training set only), so Part B models are scored on the same test rows as Part A.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

RANDOM_STATE = 42
NUM_COLS = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'net_capital']
BIN_COLS = ['overtime']


def load_clean(data_dir):
    """Pool the two UCI files, drop rows with '?', strip whitespace, drop duplicates."""
    data_dir = Path(data_dir)
    df = pd.concat([pd.read_csv(data_dir / 'adult_train.csv'),
                    pd.read_csv(data_dir / 'adult_test.csv')], ignore_index=True)
    df = df.replace('?', np.nan)
    for col in df.select_dtypes(exclude='number').columns:
        df[col] = df[col].str.strip()
    df['income'] = df['income'].str.rstrip('.')
    return df.dropna().drop_duplicates().reset_index(drop=True)


def engineer(df):
    """log1p on the skewed capital columns, add net_capital and overtime, drop fnlwgt and education."""
    out = df.copy()
    net = df['capital-gain'] - df['capital-loss']
    out['capital-gain'] = np.log1p(out['capital-gain'])
    out['capital-loss'] = np.log1p(out['capital-loss'])
    out['net_capital'] = np.sign(net) * np.log1p(net.abs())
    out['overtime'] = (df['hours-per-week'] > 40).astype(int)
    return out.drop(columns=['fnlwgt', 'education'])


def split_encode_scale(df_model):
    """Stratified 80:20 split, then fit OneHotEncoder and StandardScaler on train only."""
    target_le = LabelEncoder()
    y = pd.Series(target_le.fit_transform(df_model['income']), name='income')
    X = df_model.drop(columns=['income'])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
    X_train, X_test = X_train.reset_index(drop=True), X_test.reset_index(drop=True)
    y_train, y_test = y_train.reset_index(drop=True), y_test.reset_index(drop=True)

    cat_cols = X_train.select_dtypes(exclude='number').columns.tolist()
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(X_train[cat_cols])
    scaler = StandardScaler().fit(X_train[NUM_COLS])

    def transform(part):
        cat = pd.DataFrame(ohe.transform(part[cat_cols]), columns=ohe.get_feature_names_out(cat_cols))
        num = pd.DataFrame(scaler.transform(part[NUM_COLS]), columns=NUM_COLS)
        return pd.concat([num, part[BIN_COLS].reset_index(drop=True), cat], axis=1)

    return {
        'X_train': transform(X_train), 'X_test': transform(X_test),
        'y_train': y_train, 'y_test': y_test,
        'target_le': target_le, 'ohe': ohe, 'scaler': scaler,
    }


def prepare(data_dir='../data/raw/adult'):
    """Full Review 1 pipeline in one call."""
    return split_encode_scale(engineer(load_clean(data_dir)))
