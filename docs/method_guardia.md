# GUARD IA: Correção Poisson com Dose Calibrada e Gate para Previsões Neurais de Crime

*Método e protocolo experimental — documento companheiro de `sthyb/hybrid/{glm,guardia}.py`.*

## 1. Problema e ideia

Seja $y_{it} \in \mathbb{N}_0$ a contagem de crimes na célula $i = 1,\dots,N$
no dia $t$, e $\hat{y}_{it} > 0$ a previsão pontual de um backbone neural
espaço-temporal (STGCN/SAEA, Graph-WaveNet ou STHSL) treinado nos mesmos dados.
Os backbones deixam nos resíduos $\varepsilon_{it} = y_{it} - \hat{y}_{it}$
duas estruturas exploráveis: (i) sobre/subprevisão sistemática por célula, e
(ii) dependência espaço-temporal de defasagem curta. O GUARD IA corrige ambas
com uma **regressão Poisson por célula que trata o backbone como covariável**,
e controla quanta correção espacial cada célula recebe por meio de uma
**dose** ajustada na validação e de um **gate** por célula, capaz de desligar
a correção por completo. A saída é uma previsão nativamente probabilística,
$y_{it} \sim \text{Poisson}(\hat\mu_{it})$.

## 2. GLM de calibração por célula

Para cada célula $i$, ajusta-se por máxima verossimilhança (IRLS, somente o
período de treino):

$$\log \mathbb{E}[y_{it}] \;=\; \beta_{0i} \;+\; \alpha_i \log \hat{y}_{it}
\;+\; \beta_{1i}\,\varepsilon_{i,t-1} \;+\; \beta_{2i}\,(W\varepsilon)_{i,t-1},
\qquad y_{it}\sim\text{Poisson}. \tag{1}$$

Aqui $W$ é a matriz de pesos espaciais normalizada por linha e
$(W\varepsilon)_{i,t} = \sum_j w_{ij}\varepsilon_{jt}$. Duas decisões de
projeto importam:

- **Elasticidade livre $\alpha_i$** — o backbone entra como covariável, não
  como offset fixo; $\alpha_i \neq 1$ absorve viés multiplicativo (o offset
  fixo é recuperado em $\alpha_i = 1$). As previsões usam $\log\hat y$ com
  piso em $\log(10^{-3})$ para manter o link finito.
- **Regressores apenas defasados** — ambos os termos residuais usam $t-1$;
  nenhuma informação contemporânea de vizinhos entra na Eq. (1), de modo que a
  previsão um-passo-à-frente é viável sem simultaneidade.

Células cujo GLM falha ou retorna parâmetros inválidos recorrem ao vetor de
coeficientes média-populacional $\bar\beta$ (média sobre as células
convergidas); o método exige ao menos $20$ células convergidas, caso
contrário degrada para o backbone puro.

## 3. Dose: escalando o termo espacial

O coeficiente espacial é escalado por uma dose $c \ge 0$:

$$\hat\mu_{it}(c) \;=\; \exp\!\big(\beta_{0i} + \alpha_i \log\hat y_{it}
+ \beta_{1i}\varepsilon_{i,t-1} + c\,\beta_{2i}(W\varepsilon)_{i,t-1}\big). \tag{2}$$

$c$ é varrido na grade $\mathcal{C} = \{0, 0{,}1, \dots, 2{,}0\}$ no
**período de validação**. Para cada candidato $c$ registram-se duas curvas
por célula:

$$L_i(c) = \tfrac{1}{T_{va}}\textstyle\sum_t \big(y_{it} - \tilde\mu_{it}(c)\big)^2,
\qquad
A_i(c) = \tfrac{1}{T_{va}}\textstyle\sum_t \big|\,z_{it}\,(Wz_t)_i\,\big|, \tag{3}$$

onde $\tilde\mu(c)$ é a previsão *com gate* (Seç. 4), $z_t$ é o resíduo
centrado na seção cruzada no instante $t$, e $A_i$ é o **Moran local (LISA)**
absoluto médio — uma medida por célula do aglomeramento espacial residual. A
perda quadrática em (3) segue a perda de gate configurada (MSE aqui; MAE
recupera o critério original).

## 4. Gate

Para cada célula e dose candidata, a previsão calibrada substitui o backbone
**somente se vencer na validação**:

$$s_i(c) = \mathbb{1}\!\left[\,L_i^{\text{calib}}(c) < L_i^{\text{base}}\,\right],
\qquad
\tilde\mu_{it}(c) = s_i(c)\,\hat\mu_{it}(c) + (1 - s_i(c))\,\hat y_{it}. \tag{4}$$

O gate é a rede de segurança do método: células em que a calibração não ajuda
mantêm o backbone intocado, limitando o pior caso.

## 5. Seleção da dose — três modos

1. **Global** (`guardia`): um único $c^\*$ para todas as células, o
   **joelho de Pareto** da fronteira biobjetivo agregada
   $\big(\sqrt{\bar L(c)},\, \bar A(c)\big)$ sobre a grade com gate: entre os
   candidatos não dominados, escolhe-se o que minimiza a distância euclidiana
   ao ponto ideal após normalização min–max.
2. **Por nó, só perda** (`guardia-nodec`): $c_i^\* = \arg\min_c L_i(c)$.
3. **Pareto por nó** (`guardia-lisac`): o joelho da fronteira *por célula*
   $\big(L_i(\cdot), A_i(\cdot)\big)$, com filtragem por dominância estrita e
   normalização min–max por célula sobre a grade completa; células
   degeneradas (curvas planas) herdam o $c^\*$ global.

Os três modos compartilham um único ajuste do GLM e uma única varredura; a
seleção usa apenas a validação.

## 6. Distribuições preditivas

O GUARD IA é nativamente probabilístico:
$y_{it} \sim \text{Poisson}(\tilde\mu_{it})$. Como contagens de crime são
sobredispersas, cada modo é também avaliado sob uma Binomial Negativa NB2,
$\text{Var} = \mu + \hat\alpha\mu^2$, com a dispersão $\hat\alpha$ estimada
por MLE em grade ($\alpha \in 10^{[-4,1]}$, 60 pontos log-espaçados) sobre as
**previsões de validação do próprio modo**.

## 7. Protocolo experimental

**Dados.** Contagens diárias em grades regulares: São Paulo ($N{=}1445$),
Porto Alegre ($N{=}94$), Bahía ($N{=}74$). Divisão cronológica: últimos $110$
dias = teste, $110$ anteriores = validação, restante = treino. Os backbones
são treinados em dados z-padronizados; as previsões são revertidas à escala
original e truncadas em 0.

**Anti-vazamento.** A Eq. (1) é estimada só no treino; dose e gate usam só a
validação; os regressores de teste
$\varepsilon_{i,t-1}, (W\varepsilon)_{i,t-1}$ são construídos exclusivamente
com valores passados *observados* (a fronteira val→teste usa o último dia
observado da validação). Nada do período de teste entra em qualquer etapa de
estimação ou seleção.

**Avaliação.** (i) *Probabilística:* log score discreto médio
$\text{ALS} = -\overline{\log P(Y = y_{it})}$ numa régua inteira unificada
(PMFs nativas de Poisson/NB); histogramas de PIT aleatorizado e cobertura dos
intervalos centrais de 80/95%. (ii) *Pontual:* MAE, RMSE e
$f_{\text{worse}}$ = fração de células cujo MAE piora vs. o backbone.
(iii) *Espacial:* $I$ de Moran dos resíduos de teste com $p$-valores
analíticos por passo e teste-$t$ entre passos; PAI@$k$ para
$k \in \{1, 5, 10, 25\}\%$. (iv) *Significância:* testes de Giacomini–White
(log scores) e Diebold–Mariano (erros absolutos/quadráticos) sobre os
diferenciais diários de perda, com variância HAC de Newey–West e truncamento
$\lfloor T^{1/3} \rfloor$. As doses por célula $c_i^\*$ e os gates $s_i$ são
exportados para inspeção espacial (mapas de dose, análise de sobreposição de
gates).
