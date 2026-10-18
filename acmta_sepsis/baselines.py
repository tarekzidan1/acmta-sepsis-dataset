"""Classical machine-learning and rule-based baselines (thesis §3.7.2, Table 3.4; §4.4)."""
from __future__ import annotations
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def make_lr(seed=42):
    """Regularised (L2) logistic regression with class weighting (PROVISIONAL: C = 1.0)."""
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                         LogisticRegression(C=1.0, class_weight='balanced', max_iter=3000, random_state=seed))


def make_gbm(y, seed=42, n_jobs=2):
    """Gradient-boosted trees (XGBoost; Chen & Guestrin, 2016) with class weighting (PROVISIONAL hyper-parameters)."""
    from xgboost import XGBClassifier
    pos = max(int(y.sum()), 1); neg = len(y) - pos
    return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                         min_child_weight=5, reg_lambda=1.0, scale_pos_weight=neg / pos, tree_method='hist',
                         n_jobs=n_jobs, random_state=seed, eval_metric='logloss')


def cv_oof(kind, X, y, fold, seed=42):
    """Out-of-fold predictions for patient-grouped folds (fold id per window)."""
    oof = np.full(len(y), np.nan, dtype=np.float64)
    for k in np.unique(fold[fold >= 0]):
        tr, te = fold != k, fold == k
        m = make_lr(seed) if kind == 'lr' else make_gbm(y[tr], seed)
        m.fit(X[tr], y[tr]); oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def fit_predict(kind, Xtr, ytr, Xte, seed=42):
    m = make_lr(seed) if kind == 'lr' else make_gbm(ytr, seed)
    m.fit(Xtr, ytr)
    return m, m.predict_proba(Xte)[:, 1]
