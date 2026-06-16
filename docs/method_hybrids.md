# Híbridos Residuais ST-AR em Espaços Estabilizadores de Variância

*Método e protocolo experimental — documento companheiro de
`sthyb/hybrid/{transforms,star,predictive}.py`.*

## 1. Problema e ideia

Backbones neurais espaço-temporais produzem previsões pontuais $\hat y_{it}$
de contagens diárias de crime $y_{it}$ cujos resíduos retêm (i) viés por
célula, (ii) sazonalidade semanal e (iii) heterocedasticidade ligada ao nível
da contagem. O híbrido corrige o backbone **post hoc** com uma autorregressão
espaço-temporal sazonal de baixa dimensão ajustada aos resíduos — não na
escala de contagem, mas num **espaço transformado estabilizador de
variância** — e converte o resultado numa distribuição preditiva discreta
completa, em forma fechada.

## 2. Espaços de transformação

Seja $g$ uma transformação monótona aplicada elemento a elemento. Três
espaços rodam em paralelo, com maquinário idêntico no restante:

| Espaço | $g(y)$ | justificativa |
|---|---|---|
| aditivo (nível) | $y$ | correção aditiva clássica (baseline) |
| log1p | $\log(1+y)$ | erros multiplicativos, cauda direita pesada |
| **Anscombe** | $2\sqrt{y + 3/8}$ | estabilizadora de variância para Poisson: $\text{Var}\,g(Y) \approx 1$ |

O resíduo de trabalho é $e_{it} = g(y_{it}) - g(\hat y_{it})$. Num espaço
não linear, uma correção aditiva de $e$ torna-se uma correção *multiplicativa
e dependente do estado* na escala de contagem — células grandes recebem
ajustes proporcionalmente maiores.

## 3. ST-AR sazonal no painel de resíduos

O painel de resíduos segue um AR espaço-temporal à la Shoesmith com
defasagens sazonais explícitas:

$$e_{it} \;=\; c_i \;+\; \sum_{j \in \{1,7,14\}} \phi_j\, e_{i,t-j}
\;+\; \sum_{l \in \{1,7\}} \psi_l\, (W e)_{i,t-l} \;+\; u_{it}, \tag{1}$$

onde $c_i$ é um **efeito fixo por célula** (absorve o viés sistemático),
$\phi_j, \psi_l$ são coeficientes **globais**, $W$ é a matriz de pesos
espaciais normalizada por linha, e todos os termos espaciais são
**defasados no tempo** (sem simultaneidade). Os conjuntos de defasagens
visam o ciclo semanal identificado pela FACP agrupada dos resíduos (picos em
7 e 14).

**Estimação** por OLS within-FE: remove-se a média temporal de cada célula,
empilha-se o painel e roda-se OLS agrupado para $(\phi, \psi)$;
recupera-se
$c_i = \bar e_i - \sum_j \phi_j \bar e_i^{(j)} - \sum_l \psi_l \overline{(We)}_i^{(l)}$.
Duas covariâncias dos coeficientes são mantidas: a OLS clássica e a de
**Driscoll–Kraay** (HAC sobre as somas de escore na seção cruzada, defasagem
$\lfloor T^{1/3} \rfloor$), esta última robusta à forte dependência
cross-section quando $N$ é grande.

**Variância preditiva por célula.** $\sigma_i^2$ é o quadrado médio do
resíduo within da célula, encolhido em direção à média do painel,
$\tilde\sigma_i^2 = w\,\sigma_i^2 + (1-w)\,\bar\sigma^2$ com
$w = T/(T+30)$, e piso em $10^{-6}$.

## 4. Três janelas de ajuste

A Eq. (1) é estimada em três janelas aninhadas — **treino**, **validação** e
**pooled** (treino+validação) — gerando os métodos
`hybrid-{train,val,pooled}` por espaço. A escolha é diagnosticável: os
resíduos de treino de um backbone bem ajustado são otimisticamente pequenos
(in-sample); assim, quando um **teste de Chow** rejeita a igualdade de
coeficientes entre os ajustes de treino e validação (Wald com a covariância
de Driscoll–Kraay), prefere-se o ajuste na validação; caso contrário, o
pooling compra eficiência.

## 5. Previsão e distribuição preditiva discreta

A média corrigida no espaço transformado e a previsão pontual são

$$\mu_{it} = g(\hat y_{it}) + \hat e_{it}, \qquad
\hat y^{\text{corr}}_{it} = g^{-1}(\mu_{it}) \vee 0 , \tag{2}$$

com $\hat e_{it}$ vindo da Eq. (1) usando apenas resíduos passados
*observados*. Reporta-se também a média Jensen-correta: para Anscombe
$\mathbb{E}[y] = (\mu^2 + \tilde\sigma_i^2)/4 - 3/8$; para log1p a média
lognormal $\exp(\mu + \tilde\sigma_i^2/2) - 1$ (o espaço de nível dispensa
correção).

**Distribuição.** Assume-se
$g(Y_{it}) \sim \mathcal{N}(\mu_{it}, \tilde\sigma_i^2)$ e discretiza-se em
bordas de meio-inteiro:

$$P(Y_{it} = k) \;=\; \Phi\!\Big(\tfrac{g(k + 1/2) - \mu_{it}}{\tilde\sigma_i}\Big)
- \Phi\!\Big(\tfrac{g(k - 1/2) - \mu_{it}}{\tilde\sigma_i}\Big),\quad k \ge 1, \tag{3}$$

com $P(Y_{it} = 0) = \Phi\big((g(1/2) - \mu_{it})/\tilde\sigma_i\big)$ (o
bin inferior é aberto — nunca se avalia $g$ fora do domínio). Isso coloca
todos os métodos — Gaussiano-no-transformado e os baselines
Poisson/NB/Gaussiano sobre o backbone cru — numa **única régua
probabilística inteira**, tornando os log scores diretamente comparáveis
entre famílias.

## 6. Protocolo experimental

**Dados e divisões.** Contagens diárias: São Paulo ($N{=}1445$), Porto Alegre
($N{=}94$), Bahía ($N{=}74$); divisão cronológica com os últimos $110$ dias
como teste e os $110$ anteriores como validação. Os backbones (STGCN/SAEA,
Graph-WaveNet-MSE, STHSL-MSE) ficam congelados; o híbrido nunca os retreina.

**Anti-vazamento.** A estimação usa apenas a janela de ajuste (treino,
validação ou treino+validação). As defasagens de teste na Eq. (1) vêm
exclusivamente da sequência observada val→teste; a variância
$\tilde\sigma_i^2$ usa apenas resíduos da janela de ajuste. O período de
teste nunca entra em estimação, construção de defasagens além de valores
observados, ou qualquer seleção.

**Baselines.** O backbone cru avaliado sob três distribuições com parâmetros
estimados na validação: Poisson($\hat y$); NB2 com dispersão por MLE em grade
na validação ($\alpha \in 10^{[-4,1]}$, 60 pontos); Gaussiana em nível com
variância de validação por célula (encolhida como acima). O GUARD IA
(documento companheiro) é o método de correção concorrente.

**Métricas.** *Probabilísticas:* log score discreto médio (ALS, menor =
melhor) sobre PMFs do tipo da Eq. (3), com portões de sanidade (massa total
da PMF $\ge 0{,}999$, scores finitos) que abortam a execução em caso de
falha; PIT aleatorizado $u = F(y-1) + V\,P(Y=y)$, $V \sim U(0,1)$, em 10
bins; cobertura central de 80/95%. *Pontuais:* MAE, RMSE (cru e com média
Jensen-correta) e $f_{\text{worse}}$ (fração de células degradadas vs. o
backbone). *Espaciais:* $I$ de Moran dos resíduos de teste ($p$ analítico
por passo + teste-$t$ entre passos), PAI@$\{1,5,10,25\}\%$.
*Significância:* Giacomini–White sobre os diferenciais diários de log score e
Diebold–Mariano sobre erros absolutos/quadráticos, ambos com variância HAC de
Newey–West (defasagem $\lfloor T^{1/3} \rfloor$); contrastes pareados por
janela — Anscombe vs. aditivo e Anscombe vs. log1p — isolam o efeito do
espaço de transformação. *Diagnósticos:* razão de variância residual
treino/validação, FAC agrupada nas defasagens $\{1,7,14\}$, coeficientes por
janela com erros-padrão, e o teste de Chow clássico vs. Driscoll–Kraay da
Seç. 4.
