# === Cria o Markdown e baixa no navegador ===
from google.colab import files

content = """Classificador_KNN  
Classificador KNN - Machine Learrning Prof. Claudinei

Alunos: Alberto Zilio / Roni Pereira

Vamos utilizar o dataset presente no link:  
https://archive.ics.uci.edu/dataset/10/automobile

# Relatório do Projeto — **KNN (k-Nearest Neighbors)**

## Sumário
- [Objetivo](#objetivo)
- [Descrição do Dataset](#descrição-do-dataset)
- [Pipeline Geral (Resumo)](#pipeline-geral-resumo)
- [Preparação dos Dados](#preparação-dos-dados)
  - [Leitura e Padronização](#leitura-e-padronização)
  - [Tratamento de Nulos e Duplicatas](#tratamento-de-nulos-e-duplicatas)
  - [Verificações de Consistência](#verificações-de-consistência)
- [Modelagem KNN](#modelagem-knn)
  - [Codificação/Normalização](#codificaçãonormalização)
  - [Divisão Treino/Teste (70/30)](#divisão-treinoTeste-7030)
  - [KNN Regressor vs Classifier](#knn-regressor-vs-classifier)
- [Avaliação (Baseline)](#avaliação-baseline)
  - [Métricas](#métricas)
  - [Gráficos e Interpretação](#gráficos-e-interpretação)
- [Otimização (GridSearchCV)](#otimização-gridsearchcv)
  - [Espaço de Busca](#espaço-de-busca)
  - [Melhores Parâmetros Encontrados](#melhores-parâmetros-encontrados)
  - [Resultados de Validação Cruzada e Teste](#resultados-de-validação-cruzada-e-teste)
  - [Gráfico do Melhor Modelo (Teste)](#gráfico-do-melhor-modelo-teste)
- [Conclusões e Próximos Passos](#conclusões-e-próximos-passos)
- [Reprodutibilidade](#reprodutibilidade)

---

## Objetivo
Construir um fluxo completo de **aprendizado supervisionado** usando **KNN** sobre o dataset **Automobile (UCI)** para **prever o preço do veículo** (`price`) — tarefa de **regressão**. O objetivo é **maximizar o coeficiente de determinação (R²)** e **minimizar o erro médio (RMSE)**, mantendo o processo claro e reproduzível.

## Descrição do Dataset
- **Fonte:** UCI Machine Learning Repository – Automobile  
- **Tamanho:** 205 instâncias (linhas)  
- **Atributos:** variáveis **numéricas** (ex.: `horsepower`, `engine_size`, `length`) e **categóricas** (ex.: `make`, `fuel_type`, `body_style`).  
- **Faltantes:** existem valores ausentes (originalmente marcados com `?`).

## Pipeline Geral (Resumo)
1. **Leitura e Padronização de Colunas** (snake_case).
2. **Tratamento de dados faltantes** (mediana para numéricos, moda para categóricos).
3. **Conversões específicas** (ex.: `num_of_cylinders` textual → numérico).
4. **Remoção de duplicatas**.
5. **Separação X/y, split 70/30**.
6. **Pré-processamento**: `StandardScaler` (ou `RobustScaler`) para numéricos + `OneHotEncoder` para categóricos.
7. **Treino e avaliação** com KNN Regressor.
8. **Otimização** com `GridSearchCV` (cv=5).
9. **Gráficos** explicativos (dispersão y_real vs y_pred, resíduos).

---

## Preparação dos Dados

### Leitura e Padronização
- Arquivos utilizados no Colab: `imports-85.data` + `imports-85.names`.
- Colunas renomeadas para **snake_case** (`normalized_losses`, `engine_size`, `body_style`, etc.).
- Conversão de `num_of_cylinders` (texto: *four, six, eight* …) para coluna numérica auxiliar `num_of_cylinders_num`.

### Tratamento de Nulos e Duplicatas
- **Valores faltantes (`?`)** tratados como `NaN` e imputados:
  - **Numéricos:** mediana.
  - **Categóricos:** moda.
- **Duplicatas:** removidas (nenhuma duplicata no final).
- **Checagem de esparsidade:** colunas com pouquíssimos dados válidos seriam descartadas (limiar 60%), mas mantivemos as principais.

### Verificações de Consistência
- **Total final de linhas:** 205 (como o original).
- **Nulos restantes:** 0 (após imputação).
- **Tipos coerentes:** numéricos devidamente convertidos; categóricos preservados para `OneHotEncoder`.

---

## Modelagem KNN

### Codificação/Normalização
- **Numéricos:** escalonamento com `StandardScaler` (e também testamos `RobustScaler` na otimização).
- **Categóricos:** `OneHotEncoder(handle_unknown='ignore')`, para expandir categorias em variáveis binárias.

### Divisão Treino/Teste (70/30)
- **Treino:** 70%  
- **Teste:** 30%  
- **`random_state=42`** para reprodutibilidade.

### KNN Regressor vs Classifier
- Como o alvo é **`price`** (numérico), usamos **KNeighborsRegressor**.
- (Se quiséssemos **acurácia** em classificação, poderíamos trocar o alvo para uma coluna categórica — ou *binning* do `price`.)

---

## Avaliação (Baseline)

### Métricas
No primeiro pipeline (sem tuning), números esperados (podem variar levemente dependendo da partição):
- **R² (teste):** ~0,84  
- **RMSE (teste):** ~3.2k

> *Interpretação:*  
> - **R²** próximo de 1 indica que o modelo explica boa parte da variabilidade do preço.  
> - **RMSE** (~3.200) é o erro médio em unidades de **preço**, portanto interpretável no contexto monetário.

### Gráficos e Interpretação
> Estes gráficos são gerados pelo bloco de avaliação do pipeline (seção de regressão).

1) **Dispersão — `y_real` vs `y_pred`**  
   - **O que ver:** pontos próximos da linha tracejada `y=x` indicam boa previsão.  
   - **Como ler:** se muitos pontos estiverem **abaixo** da linha, o modelo está **superestimando**; se estiverem **acima**, **subestimando**.

2) **Histograma dos Resíduos (`y_real - y_pred`)**  
   - **O que ver:** distribuição **simétrica** e **centrada em zero** é desejável.  
   - **Como ler:** caudas longas/outliers podem indicar **pontos difíceis** ou **não linearidades**.

3) **Resíduos vs Predição**  
   - **O que ver:** um “ruído” homogêneo ao redor de 0 (sem padrão) sugere **homocedasticidade**.  
   - **Como ler:** padrões (ex.: funil) indicam heterocedasticidade — o erro cresce/decresce com o valor previsto.

---

## Otimização (GridSearchCV)

### Espaço de Busca
- **Escalonamento (numéricos):** `StandardScaler` **e** `RobustScaler`.  
- **KNN:**  
  - `n_neighbors` ∈ {3, 5, 7, 9, 11, 15, 21}  
  - `weights` ∈ {`uniform`, `distance`}  
  - `metric` = `minkowski` com `p` ∈ {1 (Manhattan), 2 (Euclidiana)}  
  - `algorithm` = `auto` (mais estável)
  - `leaf_size` ∈ {20, 30, 40}
- **Scorers (CV):** `r2` e `neg_root_mean_squared_error` (RMSE).

### Melhores Parâmetros Encontrados
> Com base em execução representativa (alvo original, sem log1p):

- **Pré-processamento:** `StandardScaler`  
- **`n_neighbors`:** **7**  
- **`weights`:** **distance**  
- **`p`:** **1** (distância de Manhattan)  
- **`metric`:** `minkowski`  
- **`algorithm`:** `auto`  
- **`leaf_size`:** 20  

> *Esses parâmetros foram obtidos como “melhores” com **refit por R²**.*

### Resultados de Validação Cruzada e Teste
- **CV (k=5):**  
  - **Melhor R² (CV):** ~**0,786**  
  - **Melhor RMSE (CV):** ~**3444**
- **Teste 70/30 (modelo escolhido pelo CV):**  
  - **R² (teste):** ~**0,847**  
  - **RMSE (teste):** ~**3241**

> *Interpretação:*  
> O modelo generaliza bem do CV para o conjunto de teste (R² aumenta levemente e RMSE cai). `weights='distance'` e `p=1` (Manhattan) costumam beneficiar KNN quando existem outliers e/ou quando a escala das features já está normalizada.

### Gráfico do Melhor Modelo (Teste)
> Gerado pela célula “**KNN Regressão — GridSearchCV + Seleção do melhor + Gráfico**”.

**Dispersão `y_real` vs `y_pred` (MELHOR MODELO)**  
- **Leitura:** a **proximidade dos pontos** à linha `y=x` confirma a boa aderência do modelo.  
- **Anotações no título:** exibimos **R² (teste)** e **RMSE (teste)** para contextualizar a performance.  
- **Padrões a notar:**  
  - Se houver “abanico” (maior dispersão para preços altos), pode indicar **heterocedasticidade**;  
  - Alguns outliers são esperados (carros de luxo, motores muito potentes).

---

## Conclusões e Próximos Passos
- O KNN, após tuning, atingiu **R² ~0,85** e **RMSE ~3,2k** no teste — **resultado sólido** para o dataset Automobile.  
- **Pontos fortes:** pipeline claro, tratamento de dados consistente, pré-processamento adequado e busca sistemática de hiperparâmetros.  
- **Oportunidades de melhoria:**
  1. **Engenharia de Atributos:** criar razões (ex.: `power_to_weight = horsepower / curb_weight`), interações ou *bins* de idade/porte do veículo.  
  2. **Tratamento de outliers:** `RobustScaler` já ajuda; poderíamos avaliar podas (winsorization) controladas.  
  3. **Modelos de comparação:** Regressão Linear regularizada (Ridge/Lasso), Árvores/Random Forest, Gradient Boosting (XGB/LightGBM).  
  4. **Validação adicional:** **repetir CV** com diferentes `random_state`/partições.  

---

## Reprodutibilidade
- **Ambiente:** Google Colab.  
- **Sementes:** `random_state=42` na divisão treino/teste e no KFold do `GridSearchCV`.  
- **Divisão:** 70% treino / 30% teste.  
- **Código principal incluído no notebook:**  
  - *Limpeza & CSV limpo:* `automobile_clean.csv`  
  - *Pipeline baseline:* split 70/30 + scaler/one-hot + KNN  
  - *Otimização:* `GridSearchCV` (cv=5) com *scorers* `r2` e `neg_root_mean_squared_error`  
  - *Seleção do melhor* e *gráfico final* `y_real vs y_pred`.

> **Observação**: os gráficos exibidos no Colab (dispersões e resíduos) foram **explicados** nas seções de avaliação. Ao rodar as células, eles são renderizados logo após as métrricas, facilitando a leitura e a inclusão de **prints** no relatório final.

---

### Anexo (exemplo de melhores parâmetros impressos na saída)
