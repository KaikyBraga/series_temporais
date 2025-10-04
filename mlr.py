import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict, Any

import json
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox

OUTDIR = Path("mlr_results")
OUTDIR.mkdir(parents=True, exist_ok=True)


def load_series(path: Path) -> pd.DataFrame:
    """
    Carrega a série num pandas.DataFrame

    Parameters
    ----------
    path : Path
        Caminho do arquivo

    Returns
    -------
    pd.DataFrame
        DataFrame com os dados do arquivo original.
    """
    df = pd.read_csv(path)
    df["week_dt"] = pd.to_datetime(df["week"])
    df = df.sort_values("week_dt").reset_index(drop=True)
    df["volume"] = df["volume"].astype(float)
    df["log_volume"] = np.log(df["volume"])
    df["dlog_volume"] = df["log_volume"].diff()
    return df.dropna().reset_index(drop=True)


def choose_period_stl(ts: pd.Series, candidates: List[int]) -> Tuple[int, pd.DataFrame]:
    """
    Função que escolhe a tendência e a sazonalidade com STL (Seasonal–Trend decomposition using Loess)

    Parameters
    ----------
    ts : pd.Series
        Série temporal
    candidates : List[int]
        Lista de candidato a período de sazonalidade

    Returns
    -------
    Tuple[int, pd.DataFrame]
        Tupla com o melhor período de sazonalidade juntamente com o DataFrame com os valores de 
        força de tendência e sazonalidade para cada período.
    """
    rows = []
    for m in candidates:
        res = STL(ts, period=m, robust=True).fit()
        
        # Força de tendência 
        Ft = max(0.0, 1.0 - np.var(res.resid) / np.var(res.trend + res.resid))
        
        # Força de sazonalidade
        Fs = max(0.0, 1.0 - np.var(res.resid) / np.var(res.seasonal + res.resid))
        rows.append({"period": m, "Ft": float(Ft), "Fs": float(Fs)})
        
    tab = pd.DataFrame(rows).sort_values("Fs", ascending=False).reset_index(drop=True)
    best = int(tab.loc[0, "period"])
    return best, tab


def add_stl_components(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """
    Adds STL decomposition components (trend, seasonal, and trend difference) to a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    period : int
        Seasonal period

    Returns
    -------
    pd.DataFrame
        DataFrame with
        - 'trend_stl': Trend component from STL decomposition.
        - 'season_stl': Seasonal component from STL decomposition.
        - 'dtrend_stl': First difference of the trend component.
        - 't': Time index as a float array.
    """
    res = STL(df["volume"], period=period, robust=True).fit()
    out = df.copy()
    out["trend_stl"] = pd.Series(res.trend, index=out.index)
    out["season_stl"] = pd.Series(res.seasonal, index=out.index)
    out["dtrend_stl"] = out["trend_stl"].diff()
    out["t"] = np.arange(len(out), dtype=float)
    return out.dropna().reset_index(drop=True)


def design(y: pd.Series, Xcols: List[str], df: pd.DataFrame):
    """
    Prepares the response and design matrix for multiple linear regression.

    Parameters
    ----------
    y : pd.Series
        The name of the response variable (target column) as a string.
    Xcols : List[str]
        List of column names to be used as predictors (features).
    df : pd.DataFrame
        The DataFrame containing the data.

    Returns
    -------
    yv : np.ndarray
        The response variable as a NumPy array of floats.
    Xv : np.ndarray
        The design matrix (predictors with intercept) as a NumPy array of floats.
    names : List[str]
        List of column names in the design matrix, including the intercept ('const').

    Notes
    -----
    If `Xcols` is empty, the design matrix will only include the intercept term.
    """
    yv = df[y].to_numpy(dtype=float)
    Xv = sm.add_constant(df[Xcols].to_numpy(dtype=float), has_constant="add") if Xcols else sm.add_constant(np.zeros((len(df),0)))
    names = ["const"] + Xcols
    return yv, Xv, names


def fit_ols(y: np.ndarray, X: np.ndarray):
    """
    Fits an Ordinary Least Squares (OLS) regression model.

    Parameters
    ----------
    y : np.ndarray
        The dependent variable (target values), as a 1-dimensional NumPy array.
    X : np.ndarray
        The independent variables (predictors), as a 2-dimensional NumPy array.

    Returns
    -------
    res : RegressionResults
        The fitted OLS regression results object from statsmodels.

    Notes
    -----
    Rows with missing values in `y` or `X` are dropped before fitting the model.
    """
    model = sm.OLS(y, X, missing="drop")
    res = model.fit()
    return res


def loocv_mse(y: np.ndarray, X: np.ndarray, res) -> float:
    """
    Compute the Leave-One-Out Cross-Validation Mean Squared Error (LOOCV MSE) for a linear regression model.

    Parameters
    ----------
    y : np.ndarray
        The observed target values, shape (n_samples,).
    X : np.ndarray
        The design matrix (features), shape (n_samples, n_features).
    res : object
        The fitted regression result object, expected to have a `fittedvalues` attribute.

    Returns
    -------
    float
        The LOOCV mean squared error.

    Notes
    -----
    This function uses the hat matrix diagonal to efficiently compute LOOCV MSE without refitting the model n times.
    """
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    H_diag = np.einsum('ij,jk,ik->i', X, XtX_inv, X)
    resid = y - res.fittedvalues
    return float(np.mean((resid / (1.0 - H_diag))**2))


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression error metrics between true and predicted values.

    Parameters
    ----------
    y_true : np.ndarray
        Array of true target values.
    y_pred : np.ndarray
        Array of predicted target values.

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
            - 'RMSE': Root Mean Squared Error
            - 'MAE': Mean Absolute Error
    """
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return {"RMSE": rmse, "MAE": mae}


def make_payload_entry(nome: str,
                       y_true: np.ndarray,
                       y_pred: np.ndarray,
                       ci_pairs: List[Tuple[float, float]]) -> Dict[str, Any]:
    return {
        "nome_modelo": nome,
        "y_true": [float(v) for v in y_true],
        "y_pred": [float(v) for v in y_pred],
        "y_dist": [(float(a), float(b)) for a, b in ci_pairs],
    }


# 1) Load + transform
df0 = load_series(Path("data.csv"))

# 2) Escolher período sazonal via STL 
period_candidates = [4, 5, 13, 26, 52, 53]
best_period, stl_table = choose_period_stl(df0["volume"], period_candidates)

# 3) Adicionar tendência e sazonalidade com STL + tendência temporal
df = add_stl_components(df0, best_period)

# 4) Split temporal 80/20
n = len(df)
n_test = max(1, int(np.floor(0.2*n)))
train = df.iloc[:-n_test].copy()
test  = df.iloc[-n_test:].copy()

# 5) Especificações de modelos (STL trend/season + tendência em t)
models = {
    "M0_intercept": [],
    "M1_time_trend": ["t"],
    "M2_stl": ["dtrend_stl", "season_stl"],
    "M3_time_plus_stl": ["t", "dtrend_stl", "season_stl"],
}

rows = []
coef_tables = {}
pred_test = test[["week_dt", "dlog_volume"]].copy()

ljung = {
    
}

for name, cols in models.items():
    y_tr, X_tr, names = design("dlog_volume", cols, train)
    res = fit_ols(y_tr, X_tr)
    y_te, X_te, _ = design("dlog_volume", cols, test)
    yhat_tr = res.predict(X_tr)
    yhat_te = res.predict(X_te)
    mtr = metrics(y_tr, yhat_tr)
    mte = metrics(y_te, yhat_te)
    r2_adj = float(res.rsquared_adj)
    loocv = loocv_mse(y_tr, X_tr, res)
    aic = float(res.aic); bic = float(res.bic)
    rows.append([name, r2_adj, loocv, aic, bic, mtr["RMSE"], mtr["MAE"], mte["RMSE"], mte["MAE"]])
    coef_df = pd.DataFrame({"term": names, "coef": res.params, "pvalue": res.pvalues})
    coef_tables[name] = coef_df
    pred_test[f"yhat_{name}"] = yhat_te
    ljung[name] = acorr_ljungbox(res.resid, lags=[10], return_df=True)
    

# 6) Testes F de inclusão (significância conjunta)
res_M0 = fit_ols(*design("dlog_volume", models["M0_intercept"], train)[:2])
res_M1 = fit_ols(*design("dlog_volume", models["M1_time_trend"], train)[:2])
res_M2 = fit_ols(*design("dlog_volume", models["M2_stl"], train)[:2])
res_M3 = fit_ols(*design("dlog_volume", models["M3_time_plus_stl"], train)[:2])

print(ljung)

f_tests = {
    "time_trend_vs_intercept": {
        "F": float(res_M1.compare_f_test(res_M0)[0]),
        "pvalue": float(res_M1.compare_f_test(res_M0)[1]),
        "df_diff": int(res_M1.compare_f_test(res_M0)[2]),
    },
    "stl_vs_intercept": {
        "F": float(res_M2.compare_f_test(res_M0)[0]),
        "pvalue": float(res_M2.compare_f_test(res_M0)[1]),
        "df_diff": int(res_M2.compare_f_test(res_M0)[2]),
    },
    "time_plus_stl_vs_stl": {
        "F": float(res_M3.compare_f_test(res_M2)[0]),
        "pvalue": float(res_M3.compare_f_test(res_M2)[1]),
        "df_diff": int(res_M3.compare_f_test(res_M2)[2]),
    },
}

coef_paths = {}
for name, table in coef_tables.items():
    p = OUTDIR / f"{name}_coeffs_stl.csv"
    table.to_csv(p, index=False)
    coef_paths[name] = str(p)

pred_path = OUTDIR / "test_predictions.csv"
pred_test.to_csv(pred_path, index=False)

stl_table_path = OUTDIR / "stl_period_search.csv"
stl_table.to_csv(stl_table_path, index=False)

with open(OUTDIR / "ftests_stl.json", "w", encoding="utf-8") as f:
    json.dump(f_tests, f, indent=2)
    

# 7) Persistência
metrics_df = pd.DataFrame(rows, columns=["model", "R2_adj", "LOOCV_MSE", "AIC", "BIC", "train_RMSE", "train_MAE", "test_RMSE", "test_MAE"])
metrics_path = OUTDIR / "ts_regression_metrics_stl.csv"
metrics_df.to_csv(metrics_path, index=False)

# Tabela de métricas
metrics_df.to_csv("arquivo.csv", index=False)

# 8) Gráfico de teste do modelo de menor RMSE
best_name = min(models.keys(), key=lambda nm: metrics_df.loc[metrics_df["model"]==nm, "test_RMSE"].values[0])
_, Xte_best, _ = design("dlog_volume", models[best_name], test)
yhat_best = fit_ols(*design("dlog_volume", models[best_name], train)[:2]).predict(Xte_best)

fig = plt.figure(figsize=(10,4))
plt.plot(test["week_dt"].to_numpy(), test["dlog_volume"].to_numpy(), label="Δlog(volume) — real")
plt.plot(test["week_dt"].to_numpy(), yhat_best, label=f"ajuste {best_name}")
plt.title(f"Teste — Real vs Ajuste (Δlog volume) — {best_name}")
plt.xlabel("Semana"); plt.ylabel("Δlog(volume)")
plt.legend()
plot_path = OUTDIR / f"{best_name}_test_fit.png"
plt.tight_layout(); plt.savefig(plot_path, dpi=150); plt.close(fig)

# 9) Reajusta o melhor modelo (já determinado) e obtém ICs 95%
res_best = fit_ols(*design("dlog_volume", models[best_name], train)[:2])

X_tr_best = design("dlog_volume", models[best_name], train)[1]
pred_tr_sf = res_best.get_prediction(X_tr_best).summary_frame(alpha=0.05)

X_te_best = design("dlog_volume", models[best_name], test)[1]
pred_te_sf = res_best.get_prediction(X_te_best).summary_frame(alpha=0.05)

# Dados reais (pred = true; ICs degenerados em (y,y))
y_tr = train["dlog_volume"].to_numpy()
y_te = test["dlog_volume"].to_numpy()

real_train = make_payload_entry(
    nome="Dados reais (treino)",
    y_true=y_tr,
    y_pred=y_tr,
    ci_pairs=[(float(v), float(v)) for v in y_tr],
)

real_test = make_payload_entry(
    nome="Dados reais (teste)",
    y_true=y_te,
    y_pred=y_te,
    ci_pairs=[(float(v), float(v)) for v in y_te],
)

# 10) Melhor modelo (IC 95% para a média prevista)
model_name_pt = "Modelo Intercepto" if best_name == "M0_intercept" else f"Modelo {best_name}"

model_train = make_payload_entry(
    nome=f"{model_name_pt} (treino)",
    y_true=y_tr,
    y_pred=pred_tr_sf["mean"].to_numpy(),
    ci_pairs=list(map(tuple, pred_tr_sf[["mean_ci_lower", "mean_ci_upper"]].to_numpy())),
)

model_test = make_payload_entry(
    nome=f"{model_name_pt} (teste)",
    y_true=y_te,
    y_pred=pred_te_sf["mean"].to_numpy(),
    ci_pairs=list(map(tuple, pred_te_sf[["mean_ci_lower", "mean_ci_upper"]].to_numpy())),
)

lista_treino = [real_train, model_train]
lista_teste = [real_test,  model_test]

#print("#"*50)
#print(lista_treino)