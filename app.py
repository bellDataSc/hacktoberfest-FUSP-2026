import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PASTA = Path(__file__).parent
POLITICA = PASTA / "resultados" / "politica_laya.json"

st.set_page_config(page_title="Triagem com Laya", layout="wide")
st.title("Triagem de chamados: a IA decide ou uma pessoa revisa?")
st.caption("Construí esta triagem com o Laya e uma política de revisão humana que calibrei no meu notebook. bel - para o hackathon com a FLUSP")

if not POLITICA.exists():
    st.error("Ainda não encontrei resultados/politica_laya.json. Eu gero esse arquivo rodando o notebook Laya_vs_Gemma.ipynb.")
    st.stop()
politica = json.loads(POLITICA.read_text(encoding="utf-8"))
rotulos = list(politica["categorias"])


@st.cache_resource(show_spinner="Carregando o Laya...")
def carregar_router():
    from laya import Router
    router = Router(max_loaded=1)
    router.load(politica["checkpoint"])
    return router


def triar(texto):
    pergunta = {"type": "choice", "instructions": politica["instrucao"], "criteria": politica["categorias"]}
    inicio = time.perf_counter()
    resposta = carregar_router().predict(texto, {"categoria": pergunta}, model=politica["checkpoint"])
    ms = (time.perf_counter() - inicio) * 1000
    p = np.array([resposta["answers"]["categoria"]["probabilities"][c] for c in rotulos])
    z = np.log(np.clip(p, 1e-6, 1)) / politica["temperatura"]
    p = np.exp(z - z.max()) / np.exp(z - z.max()).sum()
    categoria, prob = rotulos[int(p.argmax())], float(p.max())
    automatico = categoria not in politica["revisar_sempre"] and prob >= politica["limiar"]
    return {"categoria": categoria, "probabilidade": round(prob, 3), "acao": "automatico" if automatico else "revisao",
            "latencia_ms": round(ms, 1), "probs": p}


c1, c2, c3 = st.columns(3)
c1.metric("Limiar", f"{politica['limiar']:.2f}")
c2.metric("Temperatura", f"{politica['temperatura']:.2f}")
c3.metric("Erro-alvo no ajuste", f"{politica['erro_alvo']:.0%}")

um, lote = st.tabs(["Um chamado", "Lote (CSV)"])

with um:
    texto = st.text_area("Chamado", height=90, placeholder="Não quero cancelar, só trocar o cartão da cobrança.")
    if st.button("Classificar", type="primary", disabled=not texto.strip()):
        r = triar(texto.strip())
        if r["acao"] == "automatico":
            st.success(f"Automatizo e mando para a fila **{r['categoria']}** (probabilidade {r['probabilidade']:.2f})")
        else:
            st.error(f"Mando para revisão humana, com a sugestão **{r['categoria']}** (probabilidade {r['probabilidade']:.2f})")
        st.bar_chart(pd.Series(r["probs"], index=rotulos, name="probabilidade calibrada"), horizontal=True, height=240)
        st.caption(f"{r['latencia_ms']:.0f} ms")

with lote:
    arquivo = st.file_uploader("CSV com uma coluna texto", type="csv")
    if arquivo is not None:
        entrada = pd.read_csv(arquivo)
        if "texto" not in entrada.columns:
            st.error("Preciso de uma coluna texto no CSV.")
            st.stop()
        with st.spinner(f"Estou triando {len(entrada)} chamados..."):
            saida = pd.DataFrame([{k: v for k, v in triar(str(t)).items() if k != "probs"} for t in entrada.texto])
        saida = pd.concat([entrada.reset_index(drop=True), saida], axis=1)
        auto = saida.acao == "automatico"
        m1, m2 = st.columns(2)
        m1.metric("Automatizei", f"{auto.sum()} de {len(saida)}", f"{auto.mean():.0%}", delta_color="off")
        m2.metric("Mandei para revisão", int((~auto).sum()))
        st.bar_chart(saida.groupby(["categoria", "acao"]).size().unstack(fill_value=0), stack=True, height=260)
        st.dataframe(saida, hide_index=True, use_container_width=True)
        st.download_button("Baixar a minha triagem", saida.to_csv(index=False).encode("utf-8"), "triagem.csv", "text/csv")
