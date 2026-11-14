import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols

# MATRÍCULA 2115990
MATRICULA = 2115990
random.seed(MATRICULA)
r = random.random()

T_MIN = 10.0
T_MAX = 250.0

fracciones = {
    "Axial Bajo": 0.2 * r,
    "Bajo":       0.2 * r + 0.2,
    "Centro":     0.2 * r + 0.4,
    "Alto":       0.2 * r + 0.6,
    "Axial Alto": 0.2 * r + 0.8
}

replicas_por_nivel = {
    "Axial Bajo": 5,
    "Bajo":       5,
    "Centro":     10,  
    "Alto":       5,
    "Axial Alto": 5
}

temperaturas = {
    nivel: round(T_MIN + frac * (T_MAX - T_MIN), 2)
    for nivel, frac in fracciones.items()
}

print(f"=== PARÁMETROS DEL EXPERIMENTO RSM - OPCIÓN B (Matrícula: {MATRICULA}) ===")
print(f"Valor aleatorio fijo r = {r:.6f}")
for nivel, temp in temperaturas.items():
    print(f"  {nivel:<11}: T0 = {temp:<6} | Fracción = {fracciones[nivel]:.4f} | Réplicas = {replicas_por_nivel[nivel]}")
print("=" * 75 + "\n")

def generate_cities(n: int, instance_seed: int = 42) -> list[tuple[float, float]]:
    rng = random.Random(instance_seed + n)
    return [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(n)]

def tour_length(cities: list[tuple[float, float]], route: list[int]) -> float:
    return sum(
        math.dist(cities[route[i]], cities[route[(i + 1) % len(route)]])
        for i in range(len(route))
    )

def simulated_annealing_tsp(
    cities: list[tuple[float, float]],
    T_init: float,
    run_seed: int,
    T_min: float = 1e-3,
    max_iter: int = 2000
) -> float:
    
    rng = random.Random(run_seed)
    n = len(cities)
    
    current = list(range(n))
    rng.shuffle(current)
    current_cost = tour_length(cities, current)
    best_cost = current_cost

    for step in range(1, max_iter + 1):
        T = T_init * (0.995 ** step)
        if T <= T_min:
            break

        i, k = sorted(rng.sample(range(1, n), 2))
        neighbor = current[:i] + current[i:k + 1][::-1] + current[k + 1:]
        neighbor_cost = tour_length(cities, neighbor)

        delta = neighbor_cost - current_cost
        if delta < 0 or rng.random() < math.exp(-delta / T):
            current = neighbor
            current_cost = neighbor_cost
            if current_cost < best_cost:
                best_cost = current_cost

    return best_cost

# EJECUCIÓN DE LAS 90 CORRIDAS (30 POR CADA TAMAÑO n ∈ {5, 10, 15})
tamanos = [5, 10, 15]
instancias_tsp = {n: generate_cities(n, instance_seed=100) for n in tamanos}

registros = []
corrida_id = 0

for n in tamanos:
    cities = instancias_tsp[n]
    for nivel_nombre, T0 in temperaturas.items():
        num_reps = replicas_por_nivel[nivel_nombre]
        for rep in range(num_reps):
            corrida_id += 1
            semilla_replica = MATRICULA + corrida_id * 23
            costo = simulated_annealing_tsp(cities, T_init=T0, run_seed=semilla_replica)
            
            registros.append({
                "tamano": n,
                "nivel": nivel_nombre,
                "temperatura": T0,
                "replica": rep + 1,
                "costo": costo
            })

df = pd.DataFrame(registros)

def bootstrap_median_ci(data: np.ndarray, n_boot: int = 10000, alpha: float = 0.05):
    rng = np.random.default_rng(MATRICULA)
    boot_medians = np.median(rng.choice(data, size=(n_boot, len(data)), replace=True), axis=1)
    mediana_obs = np.median(data)
    ci_lower = np.percentile(boot_medians, 100 * (alpha / 2))
    ci_upper = np.percentile(boot_medians, 100 * (1 - alpha / 2))
    return mediana_obs, ci_lower, ci_upper


fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for idx, n in enumerate(tamanos):
    df_n = df[df["tamano"] == n].copy()
    
    print(f"===========================================================================")
    print(f"ANÁLISIS RSM PARA TAMAÑO n = {n} CIUDADES (30 observaciones)")
    print(f"===========================================================================")
    
    modelo = ols('costo ~ temperatura + I(temperatura**2)', data=df_n).fit()
    print(modelo.summary().tables[1])
    
    anova_reg = sm.stats.anova_lm(modelo, typ=1)
    print("\n--- ANOVA de Regresión ---")
    print(anova_reg)
    
    beta0 = modelo.params['Intercept']
    beta1 = modelo.params['temperatura']
    beta2 = modelo.params['I(temperatura ** 2)']
    p_val_b2 = modelo.pvalues['I(temperatura ** 2)']
    
    if beta2 > 0:
        t_opt_teorico = -beta1 / (2 * beta2)
        t_opt_acotado = np.clip(t_opt_teorico, min(temperaturas.values()), max(temperaturas.values()))
        tipo_curva = "Convexa (β₂ > 0 -> Mínimo válido)"
    else:
        medias_niv = df_n.groupby("temperatura")["costo"].mean()
        t_opt_teorico = float(medias_niv.idxmin())
        t_opt_acotado = t_opt_teorico
        tipo_curva = "Plana/Cóncava (β₂ ≤ 0 -> Se elige mejor nivel evaluado)"
        
    print(f"\nCurvatura: {tipo_curva} | p-value(β₂) = {p_val_b2:.4f}")
    print(f"Temperatura Óptima Calculada T* = {t_opt_acotado:.2f} (vértice sin acotar: {t_opt_teorico:.2f})\n")
    
    print("--- Medianas Bootstrap por Nivel (IC 95%) ---")
    mejor_nivel_nombre = None
    mejor_mediana = float('inf')
    mejor_ic = (0.0, 0.0)
    mejor_temp_eval = None
    
    for nivel_nombre in ["Axial Bajo", "Bajo", "Centro", "Alto", "Axial Alto"]:
        T_val = temperaturas[nivel_nombre]
        datos_nivel = df_n[df_n["nivel"] == nivel_nombre]["costo"].values
        med, low, high = bootstrap_median_ci(datos_nivel)
        print(f"  {nivel_nombre:<11} (T0={T_val:<6}): Mediana = {med:.2f} | IC 95% = [{low:.2f}, {high:.2f}]")
        
        if med < mejor_mediana:
            mejor_mediana = med
            mejor_ic = (low, high)
            mejor_nivel_nombre = nivel_nombre
            mejor_temp_eval = T_val
    print("\n")

    # GRÁFICA DE SUPERFICIE / CURVA AJUSTADA CON INTERVALO DE CONFIANZA
    ax = axes[idx]
    
    ax.scatter(df_n["temperatura"], df_n["costo"], color='#34495e', alpha=0.55, label='Observaciones (30 corridas)')
    
    t_grid = np.linspace(min(temperaturas.values()) - 5, max(temperaturas.values()) + 5, 200)
    df_pred = pd.DataFrame({"temperatura": t_grid})
    predicciones = modelo.get_prediction(df_pred).summary_frame(alpha=0.05)
    
    ax.plot(t_grid, predicciones["mean"], color='#e74c3c', linewidth=2.5, label='Ajuste Cuadrático')
    ax.fill_between(
        t_grid,
        predicciones["mean_ci_lower"],
        predicciones["mean_ci_upper"],
        color='#e74c3c',
        alpha=0.2,
        label='IC 95% del Modelo'
    )
    
    costo_en_opt = modelo.predict(pd.DataFrame({"temperatura": [t_opt_acotado]})).iloc[0]
    ax.axvline(t_opt_acotado, color='#27ae60', linestyle='--', linewidth=1.8, label=f'Óptimo T* = {t_opt_acotado:.1f}')
    ax.scatter([t_opt_acotado], [costo_en_opt], color='#27ae60', s=80, zorder=5)
    
    ax.set_title(f"Superficie de Respuesta (n = {n} ciudades)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Temperatura Inicial ($T_0$)")
    if idx == 0:
        ax.set_ylabel("Costo del Tour TSP (Distancia)")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=8.5, loc='best')

plt.tight_layout()
plt.savefig('graphics/rsm_curvas_ajustadas.png', dpi=300)
plt.show()
