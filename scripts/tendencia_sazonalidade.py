import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf

# --------- Funções utilitárias ---------
def seasonal_strength(trend: pd.Series, seasonal: pd.Series, resid: pd.Series):
    """Medida de força (Hyndman): 0 ~ fraco, 1 ~ forte."""
    var = np.nanvar
    Fs = max(0.0, 1.0 - var(resid) / var(seasonal + resid))
    Ft = max(0.0, 1.0 - var(resid) / var(trend + resid))
    return float(Ft), float(Fs)


def run_stationarity_tests(series: pd.Series):
    """ADF (H0: não estacionária) e KPSS (nível e tendência)."""
    out = {}

    adf_stat, adf_p, *_ = adfuller(series, autolag="AIC")
    out["ADF"] = {"stat": float(adf_stat), "p_value": float(adf_p)}

    kpss_stat, kpss_p, *_ = kpss(series, regression="c", nlags="auto")
    out["KPSS(level)"] = {"stat": float(kpss_stat), "p_value": float(kpss_p)}

    kpss_stat_t, kpss_p_t, *_ = kpss(series, regression="ct", nlags="auto")
    out["KPSS(trend)"] = {"stat": float(kpss_stat_t), "p_value": float(kpss_p_t)}

    return out

# --------- Execução ---------
df = pd.read_csv("data.csv")

# Série semanal
ts = df["volume"]

# Série original
plt.figure()
ts.plot(title="Volume semanal")
plt.xlabel("Semana")
plt.ylabel("Volume")
plt.tight_layout()
# if output_dir does not exist, create it
os.makedirs("./output", exist_ok=True)
plt.savefig("./output/serie.png")
plt.show()

candidatos_saz = [4, 5, 13, 26, 52, 53]
results = []
for m in candidatos_saz:
    res = STL(ts, period=m, robust=True).fit()
    Ft, Fs = seasonal_strength(res.trend, res.seasonal, res.resid)
    results.append({"m": m, "Ft": Ft, "Fs": Fs})

results_ord = sorted(results, key=lambda d: d["Fs"], reverse=True)
print(pd.DataFrame(results_ord))
period = results_ord[0]["m"]

# Decomposição STL
stl = STL(ts, period=period, robust=True)
res = stl.fit()
trend, seasonal, resid = res.trend, res.seasonal, res.resid

# Tendência
plt.figure()
trend.plot(title="Componente de Tendência (STL)")
plt.xlabel("Semana")
plt.ylabel("Tendência")
plt.tight_layout()
plt.savefig("./output/tendencia.png")
plt.show()

# Sazonalidade
plt.figure()
seasonal.plot(title=f"Componente Sazonal (período={period})")
plt.xlabel("Semana")
plt.ylabel("Sazonalidade")
plt.tight_layout()
plt.savefig("./output/sazonalidade.png")
plt.show()

# Resíduos
plt.figure()
resid.plot(title="Resíduos (após remover tendência e sazonalidade)")
plt.xlabel("Semana")
plt.ylabel("Resíduo")
plt.tight_layout()
plt.savefig("./output/residuo.png")
plt.show()

# ACF dos resíduos
plot_acf(resid, lags=40, title="ACF dos resíduos")
plt.tight_layout()
plt.savefig("./output/acf_residuos.png")
plt.show()

# Métricas
Ft, Fs = seasonal_strength(trend, seasonal, resid)
tests = run_stationarity_tests(resid)

print("\n--- Métricas principais ---")
print(f"Período sazonal estimado: {period}")
print(f"Força da Tendência (Ft): {Ft:.3f}  (≈0 fraca, ≈1 forte)")
print(f"Força da Sazonalidade (Fs): {Fs:.3f}  (≈0 fraca, ≈1 forte)")

print("\n--- Testes de estacionariedade resíduo ---")
for name, vals in tests.items():
    if "error" in vals:
        print(f"{name}: erro -> {vals['error']}")
    else:
        print(f"{name}: estatística={vals['stat']:.3f}, p-valor={vals['p_value']:.4f}")
