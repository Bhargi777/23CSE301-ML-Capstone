"""Streamlit GUI for the 23CSE301 ML capstone (Review 2 bonus deliverable D8).

Tabs:
  1. Income prediction  - Adult / Census Income classifiers (Part B models, final model = Gradient Boosting)
  2. Gas sensor cluster - assigns a sensor measurement to a K-Means cluster and shows it on the PCA map
  3. Results            - consolidated classification and clustering metrics

Run from the repository root:  streamlit run review2/app/app.py
"""
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA

REVIEW2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REVIEW2 / 'notebooks'))
import adult_prep  # noqa: E402  (shared Review 1 / Review 2 preprocessing)

ADULT_DIR = REVIEW2 / 'data' / 'raw' / 'adult'
GAS_CSV = REVIEW2 / 'data' / 'raw' / 'gassensor' / 'gas_sensor' / 'gas_sensor.csv'
CLF_DIR = REVIEW2 / 'models' / 'classification'
CLU_DIR = REVIEW2 / 'models' / 'clustering'
RES_DIR = REVIEW2 / 'results'

CLASSIFIERS = {  # display name -> saved model file; first entry is the A3 final model
    'Gradient Boosting (final model)': 'gradient_boosting.pkl',
    'Bagging': 'bagging.pkl',
    'Random Forest': 'random_forest.pkl',
    'AdaBoost': 'adaboost.pkl',
    'MLP': 'mlp.pkl',
}
GAS_NAMES = {1: 'Ethanol', 2: 'Ethylene', 3: 'Ammonia', 4: 'Acetaldehyde', 5: 'Acetone', 6: 'Toluene'}

st.set_page_config(page_title='ML Capstone Predictor', page_icon=':bar_chart:', layout='wide')


# ---------------------------------------------------------------- cached loaders
@st.cache_resource(show_spinner='Fitting the Adult preprocessing on the training split...')
def load_adult():
    """Re-run the shared pipeline: gives the train-only encoder/scaler and the input choices."""
    clean = adult_prep.load_clean(ADULT_DIR)
    data = adult_prep.split_encode_scale(adult_prep.engineer(clean))
    edu_map = clean.groupby('education')['education-num'].first().sort_values()
    choices = {col: sorted(clean[col].unique()) for col in
               ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']}
    return data, edu_map, choices


@st.cache_resource(show_spinner='Loading model...')
def load_model(path):
    return joblib.load(path)


@st.cache_resource(show_spinner='Preparing the gas sensor data...')
def load_gas():
    df = pd.read_csv(GAS_CSV)
    scaler = joblib.load(CLU_DIR / 'clustering_scaler.pkl')
    kmeans = joblib.load(CLU_DIR / 'kmeans.pkl')
    feats = [c for c in df.columns if c.startswith('feature_')]
    X = scaler.transform(signed_log(df[feats]))
    pca = PCA(n_components=2, random_state=42).fit(X)
    makeup = pd.crosstab(kmeans.predict(X), df['class'].map(GAS_NAMES), normalize='index')
    return df, feats, scaler, kmeans, pca, pca.transform(X), kmeans.predict(X), makeup


def signed_log(frame):
    """Same transform as the clustering notebooks: compress tails, keep the sign."""
    return np.sign(frame) * np.log1p(np.abs(frame))


def encode_person(raw, data):
    """Turn one raw Adult record into the 89 model columns, using the train-fitted encoder and scaler."""
    eng = adult_prep.engineer(raw)
    ohe, scaler = data['ohe'], data['scaler']
    cat_cols = list(ohe.feature_names_in_)
    cat = pd.DataFrame(ohe.transform(eng[cat_cols]), columns=ohe.get_feature_names_out(cat_cols))
    num = pd.DataFrame(scaler.transform(eng[adult_prep.NUM_COLS]), columns=adult_prep.NUM_COLS)
    X = pd.concat([num, eng[adult_prep.BIN_COLS].reset_index(drop=True), cat], axis=1)
    return X[data['X_train'].columns]


# ---------------------------------------------------------------- page
st.title('Machine Learning Capstone — Interactive Predictor')
st.caption('23CSE301 · Adult / Census Income classification · Gas Sensor Array Drift clustering')

tab_clf, tab_clu, tab_res = st.tabs(['Income prediction', 'Gas sensor cluster', 'Results'])

# ---------------------------------------------------------------- tab 1: classification
with tab_clf:
    st.subheader('Will this person earn more than $50K a year?')
    st.write('Enter a person\'s census details. The inputs go through the same cleaning, feature engineering, '
             'one-hot encoding and scaling as in the notebooks (fitted on the training split only), then the '
             'chosen trained model predicts the income class.')
    data, edu_map, choices = load_adult()

    with st.form('person'):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input('Age', min_value=17, max_value=90, value=38)
            sex = st.selectbox('Sex', choices['sex'])
            race = st.selectbox('Race', choices['race'], index=choices['race'].index('White'))
            country = st.selectbox('Native country', choices['native-country'],
                                   index=choices['native-country'].index('United-States'))
        with c2:
            education = st.selectbox('Education', list(edu_map.index), index=list(edu_map.index).index('Bachelors'))
            workclass = st.selectbox('Workclass', choices['workclass'], index=choices['workclass'].index('Private'))
            occupation = st.selectbox('Occupation', choices['occupation'],
                                      index=choices['occupation'].index('Prof-specialty'))
            hours = st.number_input('Hours per week', min_value=1, max_value=99, value=40)
        with c3:
            marital = st.selectbox('Marital status', choices['marital-status'],
                                   index=choices['marital-status'].index('Married-civ-spouse'))
            relationship = st.selectbox('Relationship', choices['relationship'],
                                        index=choices['relationship'].index('Husband'))
            gain = st.number_input('Capital gain ($)', min_value=0, max_value=99999, value=0, step=500)
            loss = st.number_input('Capital loss ($)', min_value=0, max_value=4356, value=0, step=100)
        model_name = st.selectbox('Model', list(CLASSIFIERS), help='Gradient Boosting was selected as the final '
                                  'model in A3 (best ROC-AUC, balanced accuracy and macro F1).')
        submitted = st.form_submit_button('Predict', type='primary')

    if submitted:
        raw = pd.DataFrame([{
            'age': age, 'workclass': workclass, 'fnlwgt': 0,           # fnlwgt and education are dropped by
            'education': education, 'education-num': int(edu_map[education]),  # engineer(); kept for its schema
            'marital-status': marital, 'occupation': occupation, 'relationship': relationship, 'race': race,
            'sex': sex, 'capital-gain': gain, 'capital-loss': loss, 'hours-per-week': hours,
            'native-country': country,
        }])
        X = encode_person(raw, data)
        model = load_model(CLF_DIR / CLASSIFIERS[model_name])
        p_high = float(model.predict_proba(X)[0, 1])
        label = data['target_le'].inverse_transform([int(p_high >= 0.5)])[0]

        m1, m2 = st.columns([1, 2])
        with m1:
            if label == '>50K':
                st.success(f'Predicted income: **{label}**')
            else:
                st.info(f'Predicted income: **{label}**')
            st.metric('Probability of earning >50K', f'{p_high:.1%}')
        with m2:
            st.progress(p_high, text=f'{model_name}: P(>50K) = {p_high:.2f} (threshold 0.50)')
            st.caption('Models were trained with balanced class weights, so they lean towards predicting >50K: '
                       'they catch most high earners at the cost of some false alarms.')

        with st.expander('Compare all five Part B models on this person'):
            rows = [{'model': name, 'P(>50K)': float(load_model(CLF_DIR / f).predict_proba(X)[0, 1])}
                    for name, f in CLASSIFIERS.items()]
            comp = pd.DataFrame(rows).set_index('model')
            comp['prediction'] = np.where(comp['P(>50K)'] >= 0.5, '>50K', '<=50K')
            st.dataframe(comp.style.format({'P(>50K)': '{:.3f}'}), width='stretch')
        with st.expander('Model input after preprocessing'):
            nonzero = X.loc[0][X.loc[0] != 0]
            st.write(f'{X.shape[1]} columns; non-zero values shown (scaled numeric + active one-hot columns):')
            st.dataframe(nonzero.rename('value').to_frame(), width='stretch')

# ---------------------------------------------------------------- tab 2: clustering
with tab_clu:
    st.subheader('Which cluster does a gas-sensor measurement belong to?')
    st.write('A measurement has 128 sensor features (16 sensors x 8). It is signed-log transformed and scaled '
             'with the scaler fitted in the clustering notebook, then assigned to the nearest **K-Means** '
             'centroid (k = 4). Agglomerative clustering has no `predict` for new points, so only K-Means is '
             'used here.')
    df, feats, scaler, kmeans, pca, X_pca, all_labels, makeup = load_gas()

    source = st.radio('Measurement source', ['Pick a measurement from the dataset', 'Upload a CSV'],
                      horizontal=True)
    sample, true_gas = None, None
    if source == 'Pick a measurement from the dataset':
        idx = st.number_input('Row number', min_value=0, max_value=len(df) - 1, value=0)
        sample = df.loc[[idx], feats]
        true_gas = GAS_NAMES[int(df.loc[idx, 'class'])]
    else:
        up = st.file_uploader('CSV with the columns feature_1 ... feature_128 (one row per measurement)', type='csv')
        if up is not None:
            upload = pd.read_csv(up)
            missing = [c for c in feats if c not in upload.columns]
            if missing:
                st.error(f'{len(missing)} feature columns are missing, e.g. {missing[:3]}')
            else:
                sample = upload[feats]
                st.write(f'{len(sample)} measurement(s) loaded; showing the first.')

    if sample is not None:
        Xs = scaler.transform(signed_log(sample))
        cluster = int(kmeans.predict(Xs)[0])
        dist = kmeans.transform(Xs)[0]

        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric('K-Means cluster', cluster)
            if true_gas:
                st.caption(f'True gas (not used by the model): {true_gas}')
            mk = makeup.loc[cluster].sort_values(ascending=False)
            st.write('**Gas make-up of this cluster** (whole dataset):')
            st.dataframe(mk.rename('share').to_frame().style.format('{:.0%}'), width='stretch')
            st.write('**Distance to each centroid** (scaled units):')
            st.dataframe(pd.Series(dist, name='distance').to_frame().style.format('{:.1f}'),
                         width='stretch')
        with c2:
            fig, ax = plt.subplots(figsize=(7, 5))
            cmap = plt.get_cmap('tab10')
            for k in range(kmeans.n_clusters):
                pts = X_pca[all_labels == k]
                ax.scatter(pts[:, 0], pts[:, 1], s=4, alpha=0.35, color=cmap(k), label=f'cluster {k}')
            here = pca.transform(Xs)[0]
            ax.scatter(*here, s=260, marker='*', color='black', edgecolor='white', linewidth=1.2,
                       label='this measurement', zorder=5)
            ax.set_title('Measurement on the PCA map of all 13,910 measurements')
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} of variance)')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} of variance)')
            ax.legend(markerscale=2, fontsize=8, loc='best')
            fig.tight_layout()
            st.pyplot(fig)

# ---------------------------------------------------------------- tab 3: results
with tab_res:
    st.subheader('Classification — all ten algorithms (same test split)')
    clf = pd.read_csv(RES_DIR / 'classification' / 'all_classifiers_comparison.csv', index_col='model')
    st.dataframe(clf.style.format(precision=3), width='stretch')
    st.write('**Final model:** Gradient Boosting — best ROC-AUC (0.926), balanced accuracy and macro F1; '
             'Bagging has slightly higher accuracy and weighted F1.')

    st.subheader('Clustering — Gas Sensor Array Drift (k = 4)')
    clu = pd.read_csv(RES_DIR / 'clustering' / 'clustering_summary.csv', index_col='algorithm')
    st.dataframe(clu.style.format(precision=3), width='stretch')

    c1, c2 = st.columns(2)
    c1.image(str(RES_DIR / 'classification' / 'all_classifiers_comparison.png'),
             caption='All ten classifiers on the same test set')
    c2.image(str(RES_DIR / 'clustering' / 'pca_clusters.png'), caption='PCA view of the K-Means and Ward clusters')
