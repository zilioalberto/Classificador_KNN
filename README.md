Classificador_KNN  
Classificador KNN - Machine Learrning Prof. Claudinei

Alunos: Alberto Zilio / Roni Pereira

Vamos utilizar o dataset presente no link:  
https://archive.ics.uci.edu/dataset/10/automobile
Relatório do Projeto — KNN (k-Nearest Neighbors)
Sumário
Objetivo

Descrição do Dataset

Pipeline Geral (Resumo)

Preparação dos Dados

Leitura e Padronização

Tratamento de Nulos e Duplicatas

Verificações de Consistência

Modelagem KNN

Codificação/Normalização

Divisão Treino/Teste (70/30)

KNN Regressor vs Classifier

Avaliação (Baseline)

Métricas

Gráficos e Interpretação

Otimização (GridSearchCV)

Espaço de Busca

Melhores Parâmetros Encontrados

Resultados de Validação Cruzada e Teste

Gráfico do Melhor Modelo (Teste)

Conclusões e Próximos Passos

Reprodutibilidade

Objetivo
Construir um fluxo completo de aprendizado supervisionado usando KNN sobre o dataset Automobile (UCI) para prever o preço do veículo (price) — tarefa de regressão. O objetivo é maximizar o coeficiente de determinação (R²) e minimizar o erro médio (RMSE), mantendo o processo claro e reproduzível.

Descrição do Dataset
Fonte: UCI Machine Learning Repository – Automobile

Tamanho: 205 instâncias (linhas)

Atributos: variáveis numéricas (ex.: horsepower, engine_size, length) e categóricas (ex.: make, fuel_type, body_style).

Faltantes: existem valores ausentes (originalmente marcados com ?).

Pipeline Geral (Resumo)
Leitura e Padronização de Colunas (snake_case).

Tratamento de dados faltantes (mediana para numéricos, moda para categóricos).

Conversões específicas (ex.: num_of_cylinders textual → numérico).

Remoção de duplicatas.

Separação X/y, split 70/30.

Pré-processamento: StandardScaler (ou RobustScaler) para numéricos + OneHotEncoder para categóricos.

Treino e avaliação com KNN Regressor.

Otimização com GridSearchCV (cv=5).

Gráficos explicativos (dispersão y_real vs y_pred, resíduos).

Preparação dos Dados
Leitura e Padronização
Arquivos utilizados no Colab: imports-85.data + imports-85.names.

Colunas renomeadas para snake_case (normalized_losses, engine_size, body_style, etc.).

Conversão de num_of_cylinders (texto: four, six, eight …) para coluna numérica auxiliar num_of_cylinders_num.

Tratamento de Nulos e Duplicatas
Valores faltantes (?) tratados como NaN e imputados:

Numéricos: mediana.

Categóricos: moda.

Duplicatas: removidas (nenhuma duplicata no final).

Checagem de esparsidade: colunas com pouquíssimos dados válidos seriam descartadas (limiar 60%), mas mantivemos as principais.

Verificações de Consistência
Total final de linhas: 205 (como o original).

Nulos restantes: 0 (após imputação).

Tipos coerentes: numéricos devidamente convertidos; categóricos preservados para OneHotEncoder.

Modelagem KNN
Codificação/Normalização
Numéricos: escalonamento com StandardScaler (e também testamos RobustScaler na otimização).

Categóricos: OneHotEncoder(handle_unknown='ignore'), para expandir categorias em variáveis binárias.

Divisão Treino/Teste (70/30)
Treino: 70%

Teste: 30%

random_state=42 para reprodutibilidade.

KNN Regressor vs Classifier
Como o alvo é price (numérico), usamos KNeighborsRegressor.

(Se quiséssemos acurácia em classificação, poderíamos trocar o alvo para uma coluna categórica — ou binning do price.)

Avaliação (Baseline)
Métricas
No primeiro pipeline (sem tuning):

R² (teste): (exemple: ~0,84)

RMSE (teste): (exemplo: ~3.2k)

Interpretação:

R² próximo de 1 indica que o modelo explica boa parte da variabilidade do preço.

RMSE (~3.200) é o erro médio em unidades de preço, portanto interpretável no contexto monetário.

Gráficos e Interpretação
Estes gráficos são gerados pelo bloco de avaliação do pipeline (seção de regressão).

Dispersão — y_real vs y_pred

O que ver: pontos próximos da linha tracejada y=x indicam boa previsão.

Como ler: se muitos pontos estiverem abaixo da linha, o modelo está superestimando; se estiverem acima, subestimando.

Histograma dos Resíduos (y_real - y_pred)

O que ver: distribuição simétrica e centrada em zero é desejável.

Como ler: caudas longas/outliers podem indicar pontos difíceis ou não linearidades.

Resíduos vs Predição

O que ver: um “ruído” homogêneo ao redor de 0 (sem padrão) sugere homocedasticidade.

Como ler: padrões (ex.: funil) indicam heterocedasticidade — o erro cresce/decresce com o valor previsto.

Otimização (GridSearchCV)
Espaço de Busca
Escalonamento (numéricos): StandardScaler e RobustScaler.

KNN:

n_neighbors ∈ {3, 5, 7, 9, 11, 15, 21}

weights ∈ {uniform, distance}

metric = minkowski com p ∈ {1 (Manhattan), 2 (Euclidiana)}

algorithm = auto (mais estável)

leaf_size ∈ {20, 30, 40}

Scorers (CV): r2 e neg_root_mean_squared_error (RMSE).

Melhores Parâmetros Encontrados
Com base em uma execução representativa (alvo original, sem log1p):

Pré-processamento: StandardScaler

n_neighbors: 7

weights: distance

p: 1 (distância de Manhattan)

metric: minkowski

algorithm: auto

leaf_size: 20

Esses parâmetros foram obtidos como “melhores” com refit por R².

Resultados de Validação Cruzada e Teste
CV (k=5):

Melhor R² (CV): ~0,786

Melhor RMSE (CV): ~3444

Teste 70/30 (modelo escolhido pelo CV):

R² (teste): ~0,847

RMSE (teste): ~3241

Interpretação:
O modelo generaliza bem do CV para o conjunto de teste (R² aumenta levemente e RMSE cai). weights='distance' e p=1 (Manhattan) costumam beneficiar KNN quando existem outliers e/ou quando a escala das features já está normalizada.

Gráfico do Melhor Modelo (Teste)
Gerado pela célula “KNN Regressão — GridSearchCV + Seleção do melhor + Gráfico”.

Dispersão y_real vs y_pred (MELHOR MODELO)

Leitura: a proximidade dos pontos à linha y=x confirma a boa aderência do modelo.

Anotações no título: exibimos R² (teste) e RMSE (teste) para contextualizar a performance.

Padrões a notar:

Se houver “abanico” (maior dispersão para preços altos), pode indicar heterocedasticidade;

Alguns outliers são esperados (carros de luxo, motores muito potentes).

Conclusões e Próximos Passos
O KNN, após tuning, atingiu R² ~0,85 e RMSE ~3,2k no teste — resultado sólido para o dataset Automobile.

Pontos fortes: pipeline claro, tratamento de dados consistente, pré-processamento adequado e busca sistemática de hiperparâmetros.

Oportunidades de melhoria:

Engenharia de Atributos: criar razões (ex.: power_to_weight = horsepower / curb_weight), interações ou bins de idade/porte do veículo.

Tratamento de outliers: RobustScaler já ajuda; poderíamos avaliar podas (winsorization) controladas.

Modelos de comparação: Regressão Linear regularizada (Ridge/Lasso), Árvores/Random Forest, Gradient Boosting (XGB/LightGBM).

Validação adicional: repetir CV com diferentes random_state/partições.
