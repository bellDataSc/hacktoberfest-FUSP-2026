# Laya × Gemma: quando a IA pode decidir e quando precisa de revisão humana?

**Autora:** Isabel Gonçalves Cruz · concepção, arquitetura, desenvolvimento e apresentação
**Evento:** Hacktoberfest 2026, FLUSP / CCSL IME-USP, trilha Open-Source AI

Triagem de chamados de atendimento em português com dois modelos abertos, medida em um conjunto rotulado com negação, ambiguidade e chamados sem informação suficiente. A pergunta do estudo não é qual modelo vence. É **quais chamados podem ser decididos automaticamente e quais devem ir para uma pessoa**, com uma política explícita e números medidos.

## Por que o Laya

O [Laya](https://github.com/NandhaKishorM/laya) é um modelo de decisão estruturada com código e pesos sob **Apache 2.0**. Ele recebe um texto e uma pergunta tipada (`choice`, `score`, `noul`) e devolve uma distribuição de probabilidade numa única passada, **sem gerar texto**. Por isso a saída não precisa de parsing, e o escore vem do modelo, não de um número escrito por ele.

O próprio projeto documenta limites que este estudo testa diretamente: os checkpoints vêm sobreconfiantes, o `laya-multilingual` não tem temperatura ajustada, há viés de posição das opções e a negação não fica resolvida só com rótulos semânticos.

Para comparar, o mesmo problema roda no **Gemma 4** (Google, Apache 2.0 segundo o model card, que o notebook confere e registra), com as mesmas categorias e os mesmos chamados.

## O que tem aqui

| caminho | conteúdo |
|---|---|
| `dados/chamados_ptbr.csv` | 180 chamados sintéticos, 6 categorias, partição fixa `ajuste` (72) / `avaliacao` (108) |
| `dados/GUIA_ROTULAGEM.md` | regras de rotulagem, ficha do conjunto e limitações |
| `scripts/criar_dataset.py` | gera o CSV e a partição (semente 2026) |
| `triagem.py` | código compartilhado: classificadores, métricas, calibração, política, gráficos |
| `notebooks/01_laya_vs_gemma_triagem.ipynb` | experimento completo e reproduzível |
| `app.py` | demonstração em Streamlit: entrada → classificação → automatizar ou revisar |
| `resultados/` | saídas da execução: predições, métricas, política, gráficos, ambiente |

## Protocolo

1. **Mesma tarefa.** Seis categorias (`cobranca`, `acesso`, `tecnico`, `entrega`, `cancelamento`, `sem_informacao`) com as mesmas descrições para os dois modelos.
2. **Ajuste separado da avaliação.** Variante do prompt, temperatura e limiar de revisão são escolhidos só na partição de ajuste. A avaliação roda uma vez, com tudo congelado.
   - Laya: ordem canônica das opções × seis rotações (`option_order`) promediadas na mesma passada.
   - Gemma: zero-shot × few-shot (um exemplo por categoria, tirado do ajuste e excluído da pontuação).
3. **Métricas.** Acurácia e F1 macro com IC de 95% por bootstrap, diferença pareada e McNemar exato; recorte por tipo (direto, negação, ambíguo, insuficiente); latência p50/p95 em lote 1; pico de memória na carga e na inferência.
4. **Confiabilidade dos escores, sem misturar sinais.**
   - Laya: ECE, Brier e NLL antes e depois de uma temperatura ajustada no ajuste.
   - Gemma, probabilidade calculada: softmax do log-prob de cada rótulo dado o prompt, com as mesmas métricas.
   - Gemma, confiança escrita no JSON: tratada só como sinal ordinal (AUROC acerto × erro e curva risco-cobertura), **nunca** como probabilidade.
5. **Política de revisão humana para o Laya.** `sem_informacao` sempre vai para uma pessoa. Os demais chamados são automatizados quando a probabilidade calibrada é maior ou igual a τ, o menor limiar que mantém o erro entre os automatizados em até 10% no ajuste. Na avaliação, o notebook relata quantos chamados foram automatizados, quantos erros restaram entre eles e quantos a revisão capturou.
6. **Condições registradas.** Versões dos pacotes, checkpoint e revisão (sha) no Hugging Face, licença declarada, GPU/CPU, precisão numérica, decodificação e hash do prompt ficam em `resultados/metricas.json`.

## Como executar

**Colab (recomendado):** abra o notebook, escolha um ambiente com GPU e execute em ordem. Ele clona este repositório e instala `laya==0.4.19`, `transformers>=5` e `accelerate`. O checkpoint do Gemma sai do hardware: `gemma-4-E2B-it` em T4 ou CPU e `gemma-4-E4B-it` em GPU com 22 GB ou mais.

**Local:**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/01_laya_vs_gemma_triagem.ipynb
streamlit run app.py
```

Para testar o código sem baixar pesos, defina `MODO_SIMULACAO = True` no notebook. Nesse modo os números vêm de classificadores falsos e são gravados em `resultados/simulacao/`, separados dos resultados do estudo. O app entra em modo demonstração (e avisa na tela) enquanto `resultados/politica_laya.json` não existir.

## Resultados

Os números desta execução estão em [`resultados/resumo.md`](resultados/resumo.md), gerado pelo notebook a partir de `resultados/metricas.json`, com os gráficos de confiabilidade, risco × cobertura e matrizes de confusão na mesma pasta.

## Limitações

- Os chamados são sintéticos, curtos e limpos, e os textos e os rótulos têm o mesmo autor. Uma segunda revisão dos rótulos fortaleceria o estudo.
- 108 chamados na avaliação: diferenças de poucos pontos ficam dentro do intervalo de confiança.
- Os dois modelos rodam sem fine-tuning. O Laya documenta ganhos grandes com ajuste no domínio, o que fica como próximo passo.
- Latência e memória valem para uma máquina, em lote 1, com as precisões registradas. Servidores otimizados mudariam os dois lados.
- Temperatura e limiar vêm de 72 chamados, então são estimativas com variância alta.

## Próximos passos

Fine-tuning do Laya com os chamados de ajuste (RLCD, notebook oficial), um conjunto maior com chamados reais anonimizados e dupla rotulagem, limiares por categoria e uma avaliação do Laya em lote (`predict_batch`).

## Licenças

- Código e dados deste repositório: GPL-3.0 (arquivo `LICENSE`).
- Laya: Apache 2.0 ([NandhaKishorM/laya](https://github.com/NandhaKishorM/laya), pesos em `convaiinnovations/*`).
- Gemma 4: Apache 2.0 segundo o model card. O notebook grava a licença declarada pelo checkpoint usado. Checkpoints de outras gerações têm termos próprios.
