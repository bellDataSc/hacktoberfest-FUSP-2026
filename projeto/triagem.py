"""triagem: código compartilhado do estudo "Laya × Gemma: quando a IA pode decidir e quando
precisa de revisão humana?".

Usado pelo notebook (notebooks/01_laya_vs_gemma_triagem.ipynb) e pela interface (app.py).
Autora: Isabel Gonçalves Cruz.

Princípios do módulo
- As mesmas categorias, com as mesmas descrições, valem para os dois modelos.
- Toda decisão de ajuste (variante, temperatura, limiar) usa apenas a partição `ajuste`.
- A probabilidade do Laya, a probabilidade calculada dos rótulos no Gemma e a confiança que o
  Gemma escreve ficam em colunas separadas. A confiança escrita é tratada como sinal ordinal,
  não como probabilidade.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import platform
import re
import sys
import threading
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------------------
# Tarefa
# ---------------------------------------------------------------------------------------

CATEGORIAS: Dict[str, str] = {
    "cobranca": "faturas, boletos, pagamentos, estornos, reembolsos e taxas",
    "acesso": "login, senha, código de verificação, conta bloqueada e permissões de usuários",
    "tecnico": "erros, falhas, lentidão e travamentos do aplicativo, do site ou de integrações",
    "entrega": "pedidos, rastreio, atrasos, produtos danificados ou errados, trocas e endereço de entrega",
    "cancelamento": "pedido explícito para cancelar ou encerrar assinatura, plano, conta, contrato ou pedido",
    "sem_informacao": "o texto não traz informação suficiente para saber qual equipe deve atender",
}
ROTULOS: List[str] = list(CATEGORIAS)
INSTRUCAO = "Qual equipe deve tratar este chamado de atendimento ao cliente?"

RAIZ = Path(__file__).resolve().parent
ARQUIVO_DADOS = RAIZ / "dados" / "chamados_ptbr.csv"


def carregar_dados(caminho: Path | str = ARQUIVO_DADOS) -> pd.DataFrame:
    dados = pd.read_csv(caminho, dtype={"rotulo_alternativo": str}, keep_default_na=False)
    desconhecidas = set(dados["categoria"]) - set(ROTULOS)
    if desconhecidas:
        raise ValueError(f"categorias fora do esquema: {desconhecidas}")
    return dados


def sem_acento(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(texto).lower())
                   if unicodedata.category(c) != "Mn")


def hash_texto(texto: str) -> str:
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()[:12]


def _sincronizar():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def liberar_memoria():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ---------------------------------------------------------------------------------------
# Ambiente e memória
# ---------------------------------------------------------------------------------------

def _versao(pacote: str) -> Optional[str]:
    try:
        from importlib.metadata import version
        return version(pacote)
    except Exception:
        return None


def _cpu_nome() -> str:
    try:
        for linha in Path("/proc/cpuinfo").read_text().splitlines():
            if linha.startswith("model name"):
                return linha.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def info_ambiente() -> dict:
    import psutil
    info = {
        "data_hora_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "sistema": platform.platform(),
        "colab": "google.colab" in sys.modules,
        "cpu": _cpu_nome(),
        "cpus_logicas": os.cpu_count(),
        "ram_total_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "pacotes": {p: _versao(p) for p in
                    ["laya", "torch", "transformers", "accelerate", "huggingface_hub", "numpy",
                     "pandas", "scikit-learn", "scipy"]},
        "gpu": None,
    }
    try:
        import torch
        if torch.cuda.is_available():
            prop = torch.cuda.get_device_properties(0)
            info["gpu"] = {
                "nome": prop.name,
                "vram_gb": round(prop.total_memory / 2**30, 1),
                "capacidade": f"{prop.major}.{prop.minor}",
                "cuda": torch.version.cuda,
                "n_gpus": torch.cuda.device_count(),
            }
        info["torch_threads"] = torch.get_num_threads()
    except ImportError:
        pass
    return info


class MedidorMemoria:
    """Mede o pico de memória de um trecho de código.

    - CPU: amostra o RSS do processo a cada `intervalo` segundos. O pico relatado é o maior RSS
      observado durante o trecho; `rss_delta_mb` é esse pico menos o RSS da entrada.
    - GPU: zera e lê `torch.cuda.max_memory_allocated` (tensores do PyTorch; não inclui o
      overhead do contexto CUDA nem a memória reservada pelo alocador).
    """

    def __init__(self, intervalo: float = 0.02):
        self.intervalo = intervalo
        self.resultado: dict = {}

    def _amostrar(self):
        while not self._parar.is_set():
            self._pico = max(self._pico, self._proc.memory_info().rss)
            time.sleep(self.intervalo)

    def __enter__(self):
        import psutil
        self._proc = psutil.Process()
        self._base = self._proc.memory_info().rss
        self._pico = self._base
        self._gpu = False
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                self._gpu_base = torch.cuda.memory_allocated()
                self._gpu = True
        except ImportError:
            pass
        self._parar = threading.Event()
        self._thread = threading.Thread(target=self._amostrar, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._parar.set()
        self._thread.join()
        self._pico = max(self._pico, self._proc.memory_info().rss)
        mb = 2**20
        self.resultado = {
            "rss_base_mb": round(self._base / mb, 1),
            "rss_pico_mb": round(self._pico / mb, 1),
            "rss_delta_mb": round((self._pico - self._base) / mb, 1),
        }
        if self._gpu:
            import torch
            torch.cuda.synchronize()
            self.resultado.update({
                "gpu_base_mb": round(self._gpu_base / mb, 1),
                "gpu_pico_mb": round(torch.cuda.max_memory_allocated() / mb, 1),
                "gpu_reservada_pico_mb": round(torch.cuda.max_memory_reserved() / mb, 1),
            })
        return False


def info_modelo_hub(repo_id: str) -> dict:
    """Licença declarada no model card, revisão (sha) e se o repositório exige aceite de termos."""
    try:
        from huggingface_hub import model_info
        mi = model_info(repo_id)
        card = getattr(mi, "card_data", None) or getattr(mi, "cardData", None) or {}
        licenca = card.get("license") if hasattr(card, "get") else getattr(card, "license", None)
        return {"repo": repo_id, "sha": mi.sha, "licenca_model_card": licenca,
                "gated": getattr(mi, "gated", None),
                "ultima_modificacao": str(getattr(mi, "last_modified", None))}
    except Exception as erro:  # sem rede, repositório privado etc.
        return {"repo": repo_id, "erro": f"{type(erro).__name__}: {erro}"}


# ---------------------------------------------------------------------------------------
# Laya
# ---------------------------------------------------------------------------------------

def pergunta_laya() -> dict:
    return {"type": "choice", "instructions": INSTRUCAO, "criteria": dict(CATEGORIAS)}


def perguntas_laya(variante: str) -> dict:
    """`canonica`: uma pergunta na ordem das categorias.
    `rotacoes`: k perguntas, cada uma com as opções rotacionadas (`option_order`), para que cada
    categoria ocupe cada posição uma vez. O Laya responde todas na mesma passada e as
    probabilidades são promediadas. É a mitigação de viés de posição descrita no README do Laya."""
    q = pergunta_laya()
    k = len(ROTULOS)
    if variante == "canonica":
        return {"q0": q}
    if variante == "rotacoes":
        return {f"q{r}": dict(q, option_order=[(i + r) % k for i in range(k)]) for r in range(k)}
    raise ValueError(f"variante desconhecida: {variante}")


class LayaClassificador:
    nome = "laya"

    def __init__(self, checkpoint: str = "multilingual", device: Optional[str] = None):
        from laya import Router
        self.checkpoint = checkpoint
        self.router = Router(device=device, max_loaded=1)
        self.agent = self.router.load(checkpoint)

    def classificar(self, texto: str, variante: str = "canonica") -> dict:
        perguntas = perguntas_laya(variante)
        _sincronizar()
        t0 = time.perf_counter()
        r = self.router.predict(texto, perguntas, model=self.checkpoint)
        _sincronizar()
        latencia = time.perf_counter() - t0
        matriz = np.array([[r["answers"][q]["probabilities"][c] for c in ROTULOS] for q in perguntas])
        probs = matriz.mean(axis=0)
        probs = probs / probs.sum()
        return {
            "pred": ROTULOS[int(probs.argmax())],
            "probs": probs,
            "latencia_s": latencia,
            "roteamento": r["routing"]["model"],
            # Discordância entre rotações: quantas posições diferentes levam a respostas diferentes
            "rotacoes_divergentes": int(len(set(matriz.argmax(axis=1))) > 1),
        }

    def metadados(self) -> dict:
        import laya
        a = self.agent
        cfg = getattr(a, "cfg", {}) or {}
        return {
            "laya_versao": laya.__version__,
            "checkpoint": self.checkpoint,
            "device": str(getattr(a, "device", None)),
            "dtype_autocast": str(getattr(a, "dtype", None)),
            "max_len": cfg.get("max_len"),
            "head_max_len": cfg.get("head_max_len"),
            "temperatura_checkpoint": [float(x) for x in getattr(a, "temperature", [])],
            "temperatura_por_opcoes": dict(getattr(a, "temperature_by_options", {}) or {}),
            "LAYA_CUDA_AMP": os.environ.get("LAYA_CUDA_AMP"),
        }


# ---------------------------------------------------------------------------------------
# Gemma
# ---------------------------------------------------------------------------------------

def escolher_checkpoint_gemma(ambiente: dict) -> dict:
    """Escolhe o checkpoint pelo hardware. Gemma 4 é Apache 2.0; confira a licença registrada
    em `info_modelo_hub` antes de publicar, porque checkpoints de outras gerações têm termos
    próprios (Gemma 3: Gemma Terms of Use)."""
    gpu = ambiente.get("gpu")
    if gpu and gpu["vram_gb"] >= 22:
        return {"modelo": "google/gemma-4-E4B-it", "motivo": f"GPU com {gpu['vram_gb']} GB"}
    if gpu:
        return {"modelo": "google/gemma-4-E2B-it", "motivo": f"GPU com {gpu['vram_gb']} GB (< 22 GB)"}
    return {"modelo": "google/gemma-4-E2B-it", "motivo": "sem GPU; execução em CPU será lenta"}


def dtype_gemma(preferido: str = "auto"):
    import torch
    if preferido != "auto":
        return getattr(torch, preferido)
    if torch.cuda.is_available():
        maior, _ = torch.cuda.get_device_capability()
        return torch.bfloat16 if maior >= 8 else torch.float16  # T4 (7.5) não tem bf16 nativo
    return torch.bfloat16


def prompt_sistema() -> str:
    linhas = "\n".join(f"- {c}: {d}" for c, d in CATEGORIAS.items())
    return (
        "Você faz a triagem de chamados de atendimento ao cliente.\n"
        f"{INSTRUCAO} Escolha exatamente uma categoria:\n{linhas}\n\n"
        'Responda somente com JSON, no formato {"categoria": "<chave>", "confianca": <número de 0 a 1>}, '
        "em que confianca é o quanto você está seguro da categoria."
    )


def resposta_exemplo(categoria: str) -> str:
    return json.dumps({"categoria": categoria, "confianca": 0.9}, ensure_ascii=False)


def mensagens_gemma(texto: str, exemplos: Sequence[tuple] = ()) -> List[dict]:
    msgs = [{"role": "system", "content": prompt_sistema()}]
    for ex_texto, ex_cat in exemplos:
        msgs += [{"role": "user", "content": f"Chamado: {ex_texto}"},
                 {"role": "assistant", "content": resposta_exemplo(ex_cat)}]
    msgs.append({"role": "user", "content": f"Chamado: {texto}"})
    return msgs


_SINONIMOS = {sem_acento(c).replace("_", " "): c for c in ROTULOS}
_SINONIMOS.update({"sem informacao": "sem_informacao", "sem_informacao": "sem_informacao",
                   "informacao insuficiente": "sem_informacao", "suporte tecnico": "tecnico"})


def normalizar_categoria(valor) -> Optional[str]:
    if not isinstance(valor, str):
        return None
    v = re.sub(r"[\s\-]+", " ", sem_acento(valor).strip().strip("\"'`.")).replace("_", " ")
    return _SINONIMOS.get(v)


def interpretar_resposta(texto: str) -> dict:
    """Extrai categoria e confiança declarada. status: json | recuperado | invalido."""
    limpo = re.sub(r"<think>.*?</think>", "", texto, flags=re.S)
    for bloco in re.findall(r"\{[^{}]*\}", limpo, flags=re.S):
        try:
            obj = json.loads(bloco)
        except json.JSONDecodeError:
            continue
        cat = normalizar_categoria(obj.get("categoria"))
        if cat:
            conf = obj.get("confianca", obj.get("confiança"))
            try:
                conf = float(conf)
                if 1 < conf <= 100:
                    conf /= 100
                conf = conf if 0 <= conf <= 1 else float("nan")
            except (TypeError, ValueError):
                conf = float("nan")
            return {"pred": cat, "confianca_declarada": conf, "status": "json"}
    achados = {c for c in ROTULOS if re.search(rf"\b{c}\b", sem_acento(limpo))}
    if len(achados) == 1:
        return {"pred": achados.pop(), "confianca_declarada": float("nan"), "status": "recuperado"}
    return {"pred": "invalido", "confianca_declarada": float("nan"), "status": "invalido"}


class GemmaClassificador:
    nome = "gemma"

    def __init__(self, modelo_id: str, dtype: str = "auto", max_new_tokens: int = 48,
                 enable_thinking: bool = False):
        import torch
        from transformers import AutoModelForCausalLM
        try:
            from transformers import AutoProcessor
            self.processador = AutoProcessor.from_pretrained(modelo_id)
        except Exception:
            from transformers import AutoTokenizer
            self.processador = AutoTokenizer.from_pretrained(modelo_id)
        self.tok = getattr(self.processador, "tokenizer", self.processador)
        self.dtype = dtype_gemma(dtype)
        self.modelo_id = modelo_id
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking
        self.model = AutoModelForCausalLM.from_pretrained(
            modelo_id, dtype=self.dtype, device_map="auto" if torch.cuda.is_available() else None)
        self.model.eval()
        self.device = next(self.model.parameters()).device

    # -- prompt e tokens ---------------------------------------------------------------
    def _prompt(self, mensagens: List[dict]) -> str:
        return self.processador.apply_chat_template(
            mensagens, tokenize=False, add_generation_prompt=True,
            enable_thinking=self.enable_thinking)

    def _ids(self, texto: str) -> List[int]:
        ids = self.tok(texto, add_special_tokens=False)["input_ids"]
        bos = self.tok.bos_token_id
        if bos is not None and (not ids or ids[0] != bos):
            ids = [bos] + ids
        return ids

    # -- geração (o uso típico de um LLM) -------------------------------------------------
    def gerar(self, mensagens: List[dict]) -> dict:
        import torch
        ids = torch.tensor([self._ids(self._prompt(mensagens))], device=self.device)
        _sincronizar()
        t0 = time.perf_counter()
        with torch.inference_mode():
            saida = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                        max_new_tokens=self.max_new_tokens, do_sample=False)
        _sincronizar()
        latencia = time.perf_counter() - t0
        novos = saida[0, ids.shape[1]:]
        texto = self.tok.decode(novos, skip_special_tokens=True)
        return {"resposta_bruta": texto, "tokens_gerados": int(novos.numel()),
                "tokens_entrada": int(ids.shape[1]), "latencia_s": latencia,
                **interpretar_resposta(texto)}

    # -- probabilidade calculada dos rótulos ---------------------------------------------
    def probabilidades_rotulos(self, mensagens: List[dict]) -> dict:
        """P(rótulo | prompt, '{"categoria": "') a partir dos logits, normalizada sobre as 6
        categorias. É uma probabilidade do modelo restrita ao conjunto de rótulos. Não é a
        confiança que ele escreve."""
        import torch
        prefixo = self._prompt(mensagens) + '{"categoria": "'
        pre = self._ids(prefixo)
        logps = []
        _sincronizar()
        t0 = time.perf_counter()
        with torch.inference_mode():
            for c in ROTULOS:
                completo = self._ids(prefixo + c + '"')
                base = 0
                while base < min(len(pre), len(completo)) and pre[base] == completo[base]:
                    base += 1
                n = len(completo) - base
                ids = torch.tensor([completo], device=self.device)
                try:
                    saida = self.model(input_ids=ids, logits_to_keep=n + 1).logits[0]
                except TypeError:  # versões sem logits_to_keep devolvem todas as posições
                    saida = self.model(input_ids=ids).logits[0]
                logits = saida[-(n + 1):-1]  # posições base-1 … fim-1 preveem os tokens do rótulo
                lp = torch.log_softmax(logits.float(), dim=-1)
                alvo = ids[0, base:]
                logps.append(float(lp.gather(1, alvo[:, None]).sum()))
        _sincronizar()
        lat = time.perf_counter() - t0
        z = np.array(logps)
        p = np.exp(z - z.max())
        return {"probs": p / p.sum(), "logp_rotulos": z, "latencia_prob_s": lat}

    def metadados(self) -> dict:
        return {
            "modelo": self.modelo_id,
            "dtype": str(self.dtype).replace("torch.", ""),
            "device": str(self.device),
            "decodificacao": "gulosa (do_sample=False)",
            "max_new_tokens": self.max_new_tokens,
            "enable_thinking": self.enable_thinking,
            "sha_prompt_sistema": hash_texto(prompt_sistema()),
        }


# ---------------------------------------------------------------------------------------
# Simulação: só para testar o pipeline sem baixar pesos. Resultados simulados NÃO são
# resultados do estudo; o notebook grava em resultados/simulacao/ e marca os metadados.
# ---------------------------------------------------------------------------------------

class _Simulado:
    def __init__(self, gabarito: Dict[str, str], acuracia: float, semente: int):
        self.gabarito = gabarito
        self.acuracia = acuracia
        self.semente = semente

    def _probs(self, texto: str, ajuste: int = 0) -> np.ndarray:
        rng = np.random.default_rng(int(hash_texto(f"{self.semente}{ajuste}{texto}"), 16) % 2**32)
        verdade = ROTULOS.index(self.gabarito.get(texto, "sem_informacao"))
        alvo = verdade if rng.random() < self.acuracia else int(rng.integers(len(ROTULOS)))
        z = rng.normal(0, 1, len(ROTULOS))
        z[alvo] += rng.uniform(0.5, 4.5)
        p = np.exp(z - z.max())
        return p / p.sum()


class LayaSimulado(_Simulado):
    nome = "laya"
    checkpoint = "simulado"

    def classificar(self, texto, variante="canonica"):
        t0 = time.perf_counter()
        p = self._probs(texto, 0 if variante == "canonica" else 1)
        time.sleep(0.002)
        return {"pred": ROTULOS[int(p.argmax())], "probs": p, "latencia_s": time.perf_counter() - t0,
                "roteamento": "simulado", "rotacoes_divergentes": 0}

    def metadados(self):
        return {"laya_versao": "SIMULADO", "checkpoint": "simulado"}


class GemmaSimulado(_Simulado):
    nome = "gemma"
    modelo_id = "simulado"

    def gerar(self, mensagens):
        texto = mensagens[-1]["content"].removeprefix("Chamado: ")
        p = self._probs(texto, len(mensagens))
        cat = ROTULOS[int(p.argmax())]
        bruta = json.dumps({"categoria": cat, "confianca": round(float(np.clip(p.max() + 0.2, 0, 1)), 2)})
        return {"resposta_bruta": bruta, "tokens_gerados": 14, "tokens_entrada": 300,
                "latencia_s": 0.003, **interpretar_resposta(bruta)}

    def probabilidades_rotulos(self, mensagens):
        texto = mensagens[-1]["content"].removeprefix("Chamado: ")
        p = self._probs(texto, len(mensagens))
        return {"probs": p, "logp_rotulos": np.log(p), "latencia_prob_s": 0.001}

    def metadados(self):
        return {"modelo": "SIMULADO"}


# ---------------------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------------------

def _colunas_probs(prefixo: str, probs: np.ndarray) -> dict:
    return {f"{prefixo}{c}": float(v) for c, v in zip(ROTULOS, probs)}


def executar_laya(clf, dados: pd.DataFrame, variante: str, aquecimento: int = 3) -> pd.DataFrame:
    for texto in dados["texto"].head(aquecimento):
        clf.classificar(texto, variante)
    linhas = []
    for _, row in dados.iterrows():
        r = clf.classificar(row["texto"], variante)
        linhas.append({"id": row["id"], "modelo": "laya", "variante": variante, "pred": r["pred"],
                       "p_max": float(r["probs"].max()), **_colunas_probs("p_", r["probs"]),
                       "latencia_s": r["latencia_s"], "roteamento": r["roteamento"],
                       "rotacoes_divergentes": r["rotacoes_divergentes"]})
    return pd.DataFrame(linhas)


def executar_gemma(clf, dados: pd.DataFrame, variante: str, exemplos: Sequence[tuple] = (),
                   calcular_probs: bool = True, aquecimento: int = 3) -> pd.DataFrame:
    for texto in dados["texto"].head(aquecimento):
        clf.gerar(mensagens_gemma(texto, exemplos))
    linhas = []
    for _, row in dados.iterrows():
        msgs = mensagens_gemma(row["texto"], exemplos)
        g = clf.gerar(msgs)
        linha = {"id": row["id"], "modelo": "gemma", "variante": variante, "pred": g["pred"],
                 "status_saida": g["status"], "confianca_declarada": g["confianca_declarada"],
                 "resposta_bruta": g["resposta_bruta"], "tokens_entrada": g["tokens_entrada"],
                 "tokens_gerados": g["tokens_gerados"], "latencia_s": g["latencia_s"]}
        if calcular_probs:
            pr = clf.probabilidades_rotulos(msgs)
            linha.update({"pred_prob": ROTULOS[int(pr["probs"].argmax())],
                          "p_max": float(pr["probs"].max()), **_colunas_probs("p_", pr["probs"]),
                          "latencia_prob_s": pr["latencia_prob_s"]})
        linhas.append(linha)
    return pd.DataFrame(linhas)


def matriz_probs(pred: pd.DataFrame) -> np.ndarray:
    return pred[[f"p_{c}" for c in ROTULOS]].to_numpy(dtype=float)


# ---------------------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------------------

def acuracia(y, p) -> float:
    return float(np.mean(np.asarray(y) == np.asarray(p)))


def f1_macro(y, p) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y, p, labels=ROTULOS, average="macro", zero_division=0))


def bootstrap(y, p, fn: Callable, n: int = 2000, semente: int = 0) -> tuple:
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(semente)
    valores = [fn(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return float(np.percentile(valores, 2.5)), float(np.percentile(valores, 97.5))


def bootstrap_diferenca(y, pa, pb, fn: Callable, n: int = 2000, semente: int = 0) -> dict:
    """Diferença pareada fn(a) - fn(b), reamostrando os mesmos chamados para os dois modelos."""
    y, pa, pb = map(np.asarray, (y, pa, pb))
    rng = np.random.default_rng(semente)
    difs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        difs.append(fn(y[i], pa[i]) - fn(y[i], pb[i]))
    return {"diferenca": fn(y, pa) - fn(y, pb),
            "ic95": (float(np.percentile(difs, 2.5)), float(np.percentile(difs, 97.5)))}


def mcnemar_exato(acerto_a, acerto_b) -> dict:
    from scipy.stats import binomtest
    a, b = np.asarray(acerto_a, bool), np.asarray(acerto_b, bool)
    so_a, so_b = int((a & ~b).sum()), int((~a & b).sum())
    pval = 1.0 if so_a + so_b == 0 else binomtest(so_a, so_a + so_b, 0.5).pvalue
    return {"so_a_acerta": so_a, "so_b_acerta": so_b, "p_valor": float(pval)}


def resumo_classificacao(y, p, rotulo: str) -> dict:
    lo, hi = bootstrap(y, p, acuracia)
    flo, fhi = bootstrap(y, p, f1_macro)
    return {"modelo": rotulo, "n": len(y), "acuracia": acuracia(y, p), "acuracia_ic95": (lo, hi),
            "f1_macro": f1_macro(y, p), "f1_macro_ic95": (flo, fhi)}


def resumo_latencia(segundos: Iterable[float]) -> dict:
    s = np.asarray(list(segundos)) * 1000
    return {"p50_ms": float(np.percentile(s, 50)), "p95_ms": float(np.percentile(s, 95)),
            "media_ms": float(s.mean()), "n": int(s.size)}


# ---------------------------------------------------------------------------------------
# Calibração
# ---------------------------------------------------------------------------------------

def indices(y: Sequence[str]) -> np.ndarray:
    return np.array([ROTULOS.index(c) for c in y])


def ece(conf, acerto, n_bins: int = 10) -> float:
    """Expected Calibration Error com bins de largura igual sobre a confiança do rótulo escolhido."""
    conf, acerto = np.asarray(conf, float), np.asarray(acerto, float)
    bordas = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        m = (conf > bordas[i]) & (conf <= bordas[i + 1]) if i else (conf >= 0) & (conf <= bordas[1])
        if m.any():
            total += m.mean() * abs(acerto[m].mean() - conf[m].mean())
    return float(total)


def brier(P: np.ndarray, y_idx: np.ndarray) -> float:
    um = np.eye(P.shape[1])[y_idx]
    return float(np.mean(np.sum((P - um) ** 2, axis=1)))


def nll(P: np.ndarray, y_idx: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y_idx)), y_idx], 1e-6, 1))))


def aplicar_temperatura(P: np.ndarray, T: float) -> np.ndarray:
    """Reescala log-probabilidades: softmax(log p / T). As probabilidades do Laya vêm
    arredondadas em 4 casas, por isso o piso de 1e-6."""
    z = np.log(np.clip(np.asarray(P, float), 1e-6, 1)) / T
    z -= z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def ajustar_temperatura(P: np.ndarray, y_idx: np.ndarray) -> float:
    from scipy.optimize import minimize_scalar
    r = minimize_scalar(lambda lt: nll(aplicar_temperatura(P, math.exp(lt)), y_idx),
                        bounds=(math.log(0.05), math.log(20)), method="bounded")
    return float(math.exp(r.x))


def resumo_calibracao(P: np.ndarray, y: Sequence[str], rotulo: str) -> dict:
    yi = indices(y)
    conf = P.max(axis=1)
    acerto = P.argmax(axis=1) == yi
    return {"sinal": rotulo, "ece": ece(conf, acerto), "brier": brier(P, yi), "nll": nll(P, yi),
            "confianca_media": float(conf.mean()), "acuracia": float(acerto.mean()),
            "auroc_acerto": auroc(conf, acerto)}


def auroc(conf, acerto) -> float:
    """Quanto o sinal separa acertos de erros (0,5 = acaso). Vale para qualquer sinal ordinal,
    inclusive a confiança declarada pelo Gemma."""
    from sklearn.metrics import roc_auc_score
    conf, acerto = np.asarray(conf, float), np.asarray(acerto, bool)
    ok = ~np.isnan(conf)
    if ok.sum() < 2 or acerto[ok].all() or (~acerto[ok]).all():
        return float("nan")
    return float(roc_auc_score(acerto[ok], conf[ok]))


def curva_risco_cobertura(conf, acerto) -> pd.DataFrame:
    """Ordena por confiança decrescente e mede o erro entre os k primeiros (empates recebem a
    mesma cobertura)."""
    conf, acerto = np.asarray(conf, float), np.asarray(acerto, bool)
    d = pd.DataFrame({"conf": conf, "erro": ~acerto}).dropna().sort_values("conf", ascending=False)
    g = d.groupby("conf", sort=False)["erro"].agg(["sum", "count"]).sort_index(ascending=False)
    k = g["count"].cumsum()
    return pd.DataFrame({"limiar": g.index, "cobertura": k / len(conf),
                         "risco": g["sum"].cumsum() / k}).reset_index(drop=True)


# ---------------------------------------------------------------------------------------
# Política de revisão humana
# ---------------------------------------------------------------------------------------

@dataclass
class Politica:
    limiar: float
    temperatura: float
    variante: str
    checkpoint: str
    erro_alvo: float
    revisar_sempre: List[str] = field(default_factory=lambda: ["sem_informacao"])
    min_automatizados_ajuste: int = 10
    observacao: str = ""

    def decidir(self, probs_brutas: np.ndarray) -> dict:
        p = aplicar_temperatura(np.asarray(probs_brutas)[None, :], self.temperatura)[0]
        cat = ROTULOS[int(p.argmax())]
        conf = float(p.max())
        if cat in self.revisar_sempre:
            acao, motivo = "revisao", f"categoria '{cat}' sempre vai para uma pessoa"
        elif conf < self.limiar:
            acao, motivo = "revisao", f"probabilidade calibrada {conf:.2f} < limiar {self.limiar:.2f}"
        else:
            acao, motivo = "automatico", f"probabilidade calibrada {conf:.2f} ≥ limiar {self.limiar:.2f}"
        return {"categoria": cat, "probabilidade": conf, "probs_calibradas": p, "acao": acao, "motivo": motivo}

    def salvar(self, caminho: Path | str):
        Path(caminho).write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def carregar(cls, caminho: Path | str) -> "Politica":
        return cls(**json.loads(Path(caminho).read_text(encoding="utf-8")))


def escolher_limiar(conf, pred, y, erro_alvo: float, revisar_sempre: Sequence[str],
                    min_automatizados: int = 10) -> dict:
    """Menor limiar (maior cobertura) cuja taxa de erro entre os automatizados, na partição de
    ajuste, fica ≤ erro_alvo, com pelo menos `min_automatizados` casos automatizados."""
    conf, pred, y = np.asarray(conf, float), np.asarray(pred), np.asarray(y)
    elegivel = ~np.isin(pred, list(revisar_sempre))
    for t in np.sort(np.unique(conf[elegivel])):
        auto = elegivel & (conf >= t)
        if auto.sum() < min_automatizados:
            break
        erro = float(np.mean(pred[auto] != y[auto]))
        if erro <= erro_alvo:
            return {"limiar": float(t), "encontrado": True, "erro_ajuste": erro,
                    "automatizados_ajuste": int(auto.sum())}
    return {"limiar": 1.01, "encontrado": False, "erro_ajuste": None, "automatizados_ajuste": 0}


def avaliar_politica(conf, pred, y, limiar: float, revisar_sempre: Sequence[str]) -> dict:
    conf, pred, y = np.asarray(conf, float), np.asarray(pred), np.asarray(y)
    auto = ~np.isin(pred, list(revisar_sempre)) & (conf >= limiar)
    erro = pred != y
    n_auto = int(auto.sum())
    return {
        "n": int(len(y)),
        "automatizados": n_auto,
        "cobertura": n_auto / len(y),
        "erros_entre_automatizados": int((auto & erro).sum()),
        "taxa_erro_automatizados": float((auto & erro).sum() / n_auto) if n_auto else float("nan"),
        "enviados_revisao": int((~auto).sum()),
        "erros_capturados_pela_revisao": int((~auto & erro).sum()),
        "erros_totais_sem_politica": int(erro.sum()),
    }


def bootstrap_politica(conf, pred, y, limiar, revisar_sempre, n: int = 2000, semente: int = 0) -> dict:
    conf, pred, y = np.asarray(conf, float), np.asarray(pred), np.asarray(y)
    rng = np.random.default_rng(semente)
    cob, err = [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        r = avaliar_politica(conf[i], pred[i], y[i], limiar, revisar_sempre)
        cob.append(r["cobertura"])
        if r["automatizados"]:
            err.append(r["taxa_erro_automatizados"])
    return {"cobertura_ic95": tuple(np.percentile(cob, [2.5, 97.5]).tolist()),
            "taxa_erro_automatizados_ic95": tuple(np.percentile(err, [2.5, 97.5]).tolist()) if err else None}


# ---------------------------------------------------------------------------------------
# Gráficos (matplotlib, estáticos)
# ---------------------------------------------------------------------------------------

COR_LAYA = "#2a78d6"
COR_GEMMA = "#eb6834"
TINTA = "#0b0b0b"
TINTA_SUAVE = "#52514e"
GRADE = "#e6e6e3"
SUPERFICIE = "#fcfcfb"


def _eixo(ax):
    ax.set_facecolor(SUPERFICIE)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(GRADE)
    ax.tick_params(colors=TINTA_SUAVE, labelsize=9)
    ax.grid(color=GRADE, linewidth=0.6)
    ax.set_axisbelow(True)


def grafico_confiabilidade(series: List[dict], n_bins: int = 10, titulo: str = ""):
    """series: [{"rotulo", "conf", "acerto", "cor", "estilo"}]. Um ponto por bin com n ≥ 3."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.2, 5.8), facecolor=SUPERFICIE)
    _eixo(ax)
    ax.plot([0, 1], [0, 1], color=TINTA_SUAVE, linewidth=1, linestyle=":", label="calibração perfeita")
    bordas = np.linspace(0, 1, n_bins + 1)
    for s in series:
        conf, acerto = np.asarray(s["conf"], float), np.asarray(s["acerto"], float)
        xs, ys, ns = [], [], []
        for i in range(n_bins):
            m = (conf > bordas[i]) & (conf <= bordas[i + 1]) if i else (conf <= bordas[1])
            if m.sum() >= 3:
                xs.append(conf[m].mean()); ys.append(acerto[m].mean()); ns.append(m.sum())
        ax.plot(xs, ys, color=s["cor"], linestyle=s.get("estilo", "-"), linewidth=2, marker="o",
                markersize=7, markeredgecolor=SUPERFICIE, markeredgewidth=2,
                label=f"{s['rotulo']} (ECE {ece(conf, acerto, n_bins):.3f})")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.set_xlabel("probabilidade do rótulo escolhido", color=TINTA_SUAVE)
    ax.set_ylabel("acurácia observada", color=TINTA_SUAVE)
    ax.set_title(titulo, loc="left", color=TINTA, fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    fig.tight_layout()
    return fig


def grafico_risco_cobertura(series: List[dict], erro_alvo: Optional[float] = None, titulo: str = ""):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.4, 5), facecolor=SUPERFICIE)
    _eixo(ax)
    for s in series:
        c = curva_risco_cobertura(s["conf"], s["acerto"])
        ax.step(c["cobertura"], c["risco"], where="post", color=s["cor"],
                linestyle=s.get("estilo", "-"), linewidth=2, label=s["rotulo"])
    if erro_alvo is not None:
        ax.axhline(erro_alvo, color=TINTA_SUAVE, linewidth=1, linestyle="--")
        ax.annotate(f"erro-alvo {erro_alvo:.0%}", (0.01, erro_alvo), xytext=(0, 4),
                    textcoords="offset points", fontsize=8.5, color=TINTA_SUAVE)
    ax.set_xlim(0, 1); ax.set_ylim(0, None)
    ax.set_xlabel("fração dos chamados automatizada (do mais para o menos confiante)", color=TINTA_SUAVE)
    ax.set_ylabel("taxa de erro entre os automatizados", color=TINTA_SUAVE)
    ax.set_title(titulo, loc="left", color=TINTA, fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    fig.tight_layout()
    return fig


def grafico_matrizes(pares: List[tuple], titulo: str = ""):
    """pares: [(rotulo, y_true, y_pred)]. Contagens absolutas; coluna 'invalido' se existir."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    fig, eixos = plt.subplots(1, len(pares), figsize=(5.6 * len(pares), 5), facecolor=SUPERFICIE)
    eixos = np.atleast_1d(eixos)
    for ax, (rot, y, p) in zip(eixos, pares):
        colunas = ROTULOS + (["invalido"] if "invalido" in set(p) else [])
        m = confusion_matrix(y, p, labels=colunas)[: len(ROTULOS)]
        ax.imshow(m, cmap="Blues", vmin=0)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if m[i, j]:
                    ax.text(j, i, m[i, j], ha="center", va="center", fontsize=9,
                            color="white" if m[i, j] > m.max() * 0.55 else TINTA)
        ax.set_xticks(range(len(colunas)), colunas, rotation=40, ha="right", fontsize=8.5)
        ax.set_yticks(range(len(ROTULOS)), ROTULOS, fontsize=8.5)
        ax.set_xlabel("previsto", color=TINTA_SUAVE); ax.set_ylabel("referência", color=TINTA_SUAVE)
        ax.set_title(rot, loc="left", color=TINTA, fontsize=11)
    if titulo:
        fig.suptitle(titulo, x=0.01, ha="left", color=TINTA)
    fig.tight_layout()
    return fig
