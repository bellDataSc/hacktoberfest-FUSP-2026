# Laya × Gemma: quando a IA pode decidir e quando precisa de revisão humana?

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bellDataSc/hacktoberfest-FUSP-2026/blob/main/Laya_vs_Gemma.ipynb)
![Hacktoberfest 2026](https://img.shields.io/badge/Hacktoberfest-2026-ff8ae2)
![FLUSP IME-USP](https://img.shields.io/badge/FLUSP-IME--USP-0b5394)
![Open-Source AI](https://img.shields.io/badge/trilha-Open--Source%20AI-1baf7a)
![Laya](https://img.shields.io/badge/Laya-0.4.19%20%C2%B7%20Apache%202.0-2a78d6)
![Gemma 4](https://img.shields.io/badge/Gemma%204-E2B--it%20%C2%B7%20Apache%202.0-eb6834)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit&logoColor=white)

<p align="center">
  <img src="resultados/destaque.png" width="100%" alt="Acurácia, latência e política de revisão humana do Laya e do Gemma">
</p>

Coloquei dois modelos abertos para triar os mesmos 180 chamados de atendimento em português, com negação, ambiguidade e chamados sem informação. Quero saber quais chamados a IA pode decidir sozinha e quais precisam de uma pessoa, e respondo com números que eu medi.

------------------------------------------
Ps: o Gemma acerta mais e tem escores mais úteis. O Laya é cerca de 70 vezes mais rápido e usa cerca de 6,5 vezes menos memória. Isso foi medido em chamados sintéticos e curtos.
--------------------------------------------


```mermaid
flowchart LR
    A["Chamado"] --> B["Laya<br/>uma passada, sem gerar texto"]
    B --> C{"sem_informacao?"}
    C -- sim --> R["Revisão humana"]
    C -- não --> D{"probabilidade calibrada ≥ τ?"}
    D -- sim --> F["Fila automática"]
    D -- não --> R
```

## Por que escolhi o Laya

- O Laya tem código e pesos abertos (Apache 2.0) e me devolve uma distribuição de probabilidade numa única passada, sem gerar texto.
- Trato a "confiança" que um LLM escreve como texto. Por isso calculo a probabilidade do Gemma pelos logits e comparo os dois sinais separadamente.
- Não presumo que os escores do Laya estejam calibrados: ajusto uma temperatura na partição de ajuste e meço na de avaliação.

## Limitações

- Escrevi e rotulei os chamados eu mesma; eles são sintéticos e curtos.
- Com 108 chamados na avaliação, diferenças pequenas ficam dentro do intervalo de confiança.
- Rodei os dois modelos sem fine-tuning, um chamado por vez, numa única GPU.

## Licenças

Publico este repositório sob GPL-3.0. Uso o Laya sob Apache 2.0 ([NandhaKishorM/laya](https://github.com/NandhaKishorM/laya)) e o Gemma 4 sob Apache 2.0, conforme o model card; registro a licença do checkpoint que usei em `resultados/execucao.json`.

---

bel - para o hackathon com a FLUSP
