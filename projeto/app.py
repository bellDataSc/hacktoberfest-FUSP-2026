"""Demonstração: um chamado entra, o Laya classifica e a política decide entre automatizar e
mandar para revisão humana.

    streamlit run app.py

Lê a política exportada pelo notebook (resultados/politica_laya.json). Sem esse arquivo, ou com
LAYA_DEMO_SIMULADO=1, roda em modo demonstração: ele avisa na tela e não usa o modelo real.
Autora: Isabel Gonçalves Cruz.
"""
import os
import time

import numpy as np
import pandas as pd
import streamlit as st

import triagem as tg

st.set_page_config(page_title="Laya × Gemma · triagem com revisão humana", layout="centered")

ARQ_POLITICA = tg.RAIZ / "resultados" / "politica_laya.json"
ARQ_GEMMA = tg.RAIZ / "resultados" / "predicoes_gemma.csv"
SIMULADO = os.environ.get("LAYA_DEMO_SIMULADO") == "1" or not ARQ_POLITICA.exists()


@st.cache_resource(show_spinner="Carregando o Laya…")
def carregar_classificador():
    if SIMULADO:
        dados = tg.carregar_dados()
        return tg.LayaSimulado(dict(zip(dados.texto, dados.categoria)), acuracia=0.7, semente=1)
    return tg.LayaClassificador(checkpoint=carregar_politica().checkpoint)


@st.cache_data
def carregar_politica() -> tg.Politica:
    if ARQ_POLITICA.exists() and not os.environ.get("LAYA_DEMO_SIMULADO") == "1":
        return tg.Politica.carregar(ARQ_POLITICA)
    return tg.Politica(limiar=0.80, temperatura=1.0, variante="canonica", checkpoint="multilingual",
                       erro_alvo=0.10, observacao="valores de demonstração, não ajustados")


@st.cache_data
def carregar_exemplos() -> pd.DataFrame:
    dados = tg.carregar_dados()
    return dados[dados.particao == "avaliacao"]


@st.cache_data
def carregar_gemma() -> pd.DataFrame | None:
    if not ARQ_GEMMA.exists():
        return None
    g = pd.read_csv(ARQ_GEMMA)
    return g[g.particao == "avaliacao"].set_index("id")


politica = carregar_politica()
exemplos = carregar_exemplos()

st.title("Quando a IA pode decidir sozinha?")
st.caption("Triagem de chamados com o Laya (Apache 2.0) e uma política de revisão humana · "
           "Isabel Gonçalves Cruz · Hacktoberfest 2026 FLUSP")
if SIMULADO:
    st.warning("Modo demonstração: classificador simulado e política não ajustada. "
               "Rode o notebook para gerar `resultados/politica_laya.json`.")

with st.sidebar:
    st.subheader("Política em uso")
    st.markdown(
        f"- limiar τ: **{politica.limiar:.2f}**\n"
        f"- temperatura: **{politica.temperatura:.2f}**\n"
        f"- variante: `{politica.variante}`\n"
        f"- erro-alvo no ajuste: {politica.erro_alvo:.0%}\n"
        f"- sempre revisar: {', '.join(politica.revisar_sempre) or '—'}")
    if politica.observacao:
        st.caption(politica.observacao)
    st.caption("τ e temperatura foram escolhidos só com a partição de ajuste.")

opcoes = ["(escrever o meu)"] + [f"{r.id} · {r.tipo} · {r.texto}" for r in exemplos.itertuples()]
escolha = st.selectbox("Exemplo da partição de avaliação", opcoes)
linha = None if escolha == opcoes[0] else exemplos[exemplos.id == escolha.split(" · ")[0]].iloc[0]
texto = st.text_area("Chamado", value="" if linha is None else linha.texto, height=100,
                     placeholder="Ex.: Não quero cancelar, só preciso trocar o cartão da cobrança.")

if st.button("Classificar", type="primary", disabled=not texto.strip()):
    clf = carregar_classificador()
    t0 = time.perf_counter()
    r = clf.classificar(texto.strip(), politica.variante)
    d = politica.decidir(r["probs"])
    ms = (time.perf_counter() - t0) * 1000

    if d["acao"] == "automatico":
        st.success(f"**Automatizar → fila `{d['categoria']}`**  \n{d['motivo']}")
    else:
        st.error(f"**Revisão humana** (sugestão: `{d['categoria']}`)  \n{d['motivo']}")

    col1, col2 = st.columns(2)
    col1.metric("Probabilidade calibrada", f"{d['probabilidade']:.2f}")
    col2.metric("Tempo desta chamada", f"{ms:.0f} ms")

    tabela = pd.DataFrame({"categoria": tg.ROTULOS, "probabilidade calibrada": d["probs_calibradas"],
                           "probabilidade bruta": r["probs"]}).sort_values("probabilidade calibrada", ascending=False)
    st.bar_chart(tabela.set_index("categoria")["probabilidade calibrada"], horizontal=True, height=240)
    with st.expander("Distribuição completa"):
        st.dataframe(tabela.style.format({"probabilidade calibrada": "{:.3f}", "probabilidade bruta": "{:.3f}"}),
                     hide_index=True)

    if linha is not None:
        st.markdown(f"**Rótulo de referência:** `{linha.categoria}` "
                    f"({'acertou' if d['categoria'] == linha.categoria else 'errou'}) · tipo `{linha.tipo}`")
        gemma = carregar_gemma()
        if gemma is not None and linha.id in gemma.index:
            g = gemma.loc[linha.id]
            conf = "—" if pd.isna(g.confianca_declarada) else f"{g.confianca_declarada:.2f}"
            st.markdown(f"**Gemma (registrado no notebook):** `{g.pred}` · confiança escrita {conf} "
                        f"· {g.latencia_s * 1000:.0f} ms  \n"
                        "_A confiança escrita pelo Gemma é texto gerado, não uma probabilidade calculada._")
