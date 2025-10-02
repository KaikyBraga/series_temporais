import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import lognorm
import scoringrules as sr
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# 1. CARREGAMENTO E PREPARAÇÃO DOS DADOS
print("1. Carregando e preparando os dados...")

try:
    df = pd.read_csv('data.csv')
except FileNotFoundError:
    print("Erro: O arquivo 'data.csv' não foi encontrado. Certifique-se de que ele está no mesmo diretório que o script.")
    exit()
# data para datetime
df['week'] = pd.to_datetime(df['week'])
df = df.set_index('week')
df.sort_index(inplace=True)

# Transformação para log com fim de estabilizar a variância
df['log_volume'] = np.log(df['volume'])

print(f"Dados carregados com sucesso. Período: {df.index.min().date()} a {df.index.max().date()}.")
print("-" * 50)


# 2. Treino e teste
print("2. Dividindo os dados em conjuntos de treino e teste...")

test_size = 52
horizon = test_size

if len(df) <= test_size:
    print("Erro: O conjunto de dados é muito pequeno para uma divisão de treino/teste de 52 semanas.")
    exit()

train = df.iloc[:-test_size]
test = df.iloc[-test_size:]

train_orig = train['volume']
test_orig = test['volume']
train_log = train['log_volume']

print(f"Tamanho do conjunto de treino: {len(train)} semanas")
print(f"Tamanho do conjunto de teste (horizonte): {len(test)} semanas")
print("-" * 50)


# Definição dos modelos baselines que queremos analisar

def mean_model(train_series, h):
    prediction = np.full(h, train_series.mean())
    in_sample_preds = np.full_like(train_series, train_series.mean())
    return prediction, in_sample_preds

def naive_model(train_series, h):
    prediction = np.full(h, train_series.iloc[-1])
    in_sample_preds = train_series.shift(1)
    return prediction, in_sample_preds

def seasonal_naive_model(train_series, h, m=52):
    last_season = train_series.iloc[-m:]
    prediction = np.tile(last_season, (h // m) + 1)[:h]
    in_sample_preds = train_series.shift(m)
    return prediction, in_sample_preds

def drift_model(train_series, h):
    T = len(train_series)
    drift = (train_series.iloc[-1] - train_series.iloc[0]) / (T - 1)
    prediction = train_series.iloc[-1] + np.arange(1, h + 1) * drift
    in_sample_preds = train_series.shift(1) + drift
    return prediction, in_sample_preds


# Métricas e critérios de avaliação

def calculate_all_metrics(y_true, y_pred, train_true, mu_log, sigma_log):
    metrics = {}

    metrics['MAE'] = mean_absolute_error(y_true, y_pred)
    metrics['MSE'] = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(metrics['MSE'])

    in_sample_naive_error = train_true.diff(1).dropna()
    scale_mae = np.mean(np.abs(in_sample_naive_error))
    scale_rmse = np.sqrt(np.mean(np.square(in_sample_naive_error)))
    metrics['MASE'] = metrics['MAE'] / scale_mae if scale_mae != 0 else np.inf
    metrics['RMSSE'] = rmse / scale_rmse if scale_rmse != 0 else np.inf

    dist_params = {"s": sigma_log, "scale": np.exp(mu_log)}

    try:
        n_samples = 1000
        ensemble_forecasts = lognorm.rvs(s=sigma_log, scale=np.exp(mu_log), size=(n_samples, len(y_true))).T
        crps_scores = sr.crps_ensemble(y_true.values, ensemble_forecasts)
        metrics['CRPS'] = np.mean(crps_scores)
    except AttributeError:
        print("Aviso: 'crps_ensemble' não encontrado em 'scoringrules'. A métrica CRPS será ignorada.")
        metrics['CRPS'] = np.nan
    except Exception as e:
        print(f"Ocorreu um erro ao calcular o CRPS: {e}")
        metrics['CRPS'] = np.nan

    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    avg_quantile_loss = 0
    for q in quantiles:
        pred_q = lognorm.ppf(q, **dist_params)
        errors = y_true - pred_q
        q_loss = np.mean(np.maximum(q * errors, (q - 1) * errors))
        metrics[f'QL_q{int(q*100)}'] = q_loss
        avg_quantile_loss += q_loss
    metrics['AvgQL'] = avg_quantile_loss / len(quantiles)

    alpha = 0.10
    lower_bound = lognorm.ppf(alpha / 2, **dist_params)
    upper_bound = lognorm.ppf(1 - alpha / 2, **dist_params)
    interval_width = upper_bound - lower_bound
    lower_penalty = np.where(y_true < lower_bound, (lower_bound - y_true) * (2 / alpha), 0)
    upper_penalty = np.where(y_true > upper_bound, (y_true - upper_bound) * (2 / alpha), 0)
    winkler_scores = interval_width + lower_penalty + upper_penalty
    metrics['Winkler90'] = np.mean(winkler_scores)
    
    return metrics


# Run it all 
print("3. Executando e avaliando os modelos de referência...")

models = {
    "Média": mean_model, "Naive": naive_model,
    "Naive Sazonal": seasonal_naive_model, "Drift": drift_model
}

results_list = []
predictions_dict = {}

for name, model_func in models.items():
    log_preds, log_in_sample_preds = model_func(train_log, horizon)
    predictions = np.exp(log_preds)
    predictions_dict[name] = pd.Series(predictions, index=test_orig.index)
    
    log_residuals = (train_log - log_in_sample_preds).dropna()
    sigma = np.std(log_residuals)
    
    model_metrics = calculate_all_metrics(
        y_true=test_orig, y_pred=predictions, train_true=train_orig,
        mu_log=log_preds, sigma_log=sigma
    )
    model_metrics['Modelo'] = name
    results_list.append(model_metrics)
    print(f"  - Modelo '{name}' avaliado.")

results_df = pd.DataFrame(results_list).set_index('Modelo')
point_cols = ['MAE', 'MSE', 'MASE', 'RMSSE']
dist_cols = ['CRPS', 'AvgQL', 'Winkler90']
quantile_cols = [col for col in results_df.columns if 'QL_q' in col]
results_df = results_df[point_cols + dist_cols + quantile_cols]

print("-" * 50)


# Display dos resultados encontrados
print("4. Resultados e Avaliação Comparativa:\n")
pd.set_option('display.float_format', '{:.4f}'.format)
print(results_df)
print("\nLegenda das Métricas:")
print("  - MASE/RMSSE: Erros Escalados ( <1 é melhor que o Naive no treino)")
print("  - CRPS/AvgQL/Winkler90: Métricas de previsão distribucional (menor é melhor)")
print("-" * 50)


# Viz
print("5. Gerando gráfico de previsões...")

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(15, 8))

plot_window = 104
if len(train_orig) > plot_window:
    plot_train_data = train_orig.iloc[-plot_window:]
else:
    plot_train_data = train_orig

ax.plot(plot_train_data, label='Dados de Treino', color='gray')
ax.plot(test_orig, label='Dados Reais (Teste)', color='black', linewidth=2)

colors = {'Média': 'orange', 'Naive': 'purple', 'Naive Sazonal': 'green', 'Drift': 'red'}
for name, preds in predictions_dict.items():
    ax.plot(preds, label=f'Previsão {name}', color=colors[name], linestyle='--')

snaive_log_preds, snaive_log_in_sample = seasonal_naive_model(train_log, horizon)
snaive_log_residuals = (train_log - snaive_log_in_sample).dropna()
snaive_sigma = np.std(snaive_log_residuals)
snaive_dist_params = {"s": snaive_sigma, "scale": np.exp(snaive_log_preds)}
snaive_lower = lognorm.ppf(0.05, **snaive_dist_params)
snaive_upper = lognorm.ppf(0.95, **snaive_dist_params)
ax.fill_between(test_orig.index, snaive_lower, snaive_upper, color='green', alpha=0.2,
                label='Intervalo de Predição 90% (Naive Sazonal)')

ax.set_title('Comparação de Previsões dos Modelos de Referência', fontsize=16)
ax.set_xlabel('Semana', fontsize=12)
ax.set_ylabel('Volume', fontsize=12)
ax.legend(loc='upper left')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.xticks(rotation=45)
plt.tight_layout()

print("Gráfico gerado. Exibindo...")
plt.show()
