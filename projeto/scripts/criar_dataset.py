"""Gera dados/chamados_ptbr.csv: chamados sintéticos em português, rotulados, com partição fixa.

Os textos foram escritos para este estudo; não vêm de clientes reais. Cada linha tem:
- categoria: rótulo de referência, definido pelo guia em dados/GUIA_ROTULAGEM.md
- tipo: direto | negacao | ambiguo | insuficiente
- rotulo_alternativo: segunda leitura plausível (só nos ambíguos)
- contem_negacao: marcador léxico (não, nunca, nenhum...), calculado automaticamente
- particao: ajuste (40%) | avaliacao (60%), estratificada por categoria x tipo, semente fixa

Uso:  python scripts/criar_dataset.py
"""
import csv
import random
import re
import unicodedata
from pathlib import Path

SEMENTE = 2026
FRACAO_AJUSTE = 0.40
SAIDA = Path(__file__).resolve().parents[1] / "dados" / "chamados_ptbr.csv"

# (texto, categoria, tipo, rotulo_alternativo)
CHAMADOS = [
    # ---------------- cobranca ----------------
    ("Fui cobrada duas vezes na fatura de setembro, preciso do estorno.", "cobranca", "direto", ""),
    ("O boleto deste mês veio com valor maior do que o contratado.", "cobranca", "direto", ""),
    ("Quero a segunda via da fatura de outubro.", "cobranca", "direto", ""),
    ("Paguei o Pix ontem e o sistema ainda mostra a conta em aberto.", "cobranca", "direto", ""),
    ("Apareceu uma taxa de R$ 19,90 que eu não reconheço no meu cartão.", "cobranca", "direto", ""),
    ("Como faço para mudar a data de vencimento da minha fatura?", "cobranca", "direto", ""),
    ("Preciso da nota fiscal do pagamento feito em agosto.", "cobranca", "direto", ""),
    ("O desconto prometido na renovação não foi aplicado na cobrança.", "cobranca", "direto", ""),
    ("Quero trocar o cartão de crédito cadastrado para pagamento.", "cobranca", "direto", ""),
    ("Solicito reembolso da mensalidade cobrada após o período de teste.", "cobranca", "direto", ""),
    ("A fatura veio em nome da empresa errada, preciso corrigir o CNPJ.", "cobranca", "direto", ""),
    ("Quero parcelar o valor em atraso, quais são as opções?", "cobranca", "direto", ""),

    ("Não quero cancelar o plano, só preciso que estornem a cobrança duplicada.", "cobranca", "negacao", ""),
    ("Não é problema de acesso, eu consigo entrar; o problema é que fui cobrado duas vezes.", "cobranca", "negacao", ""),
    ("O aplicativo não está com erro nenhum, mas a fatura veio com um valor que não reconheço.", "cobranca", "negacao", ""),
    ("Não recebi o reembolso que vocês prometeram há quinze dias.", "cobranca", "negacao", ""),
    ("Meu pedido chegou certinho, não tenho reclamação da entrega; quero apenas a nota fiscal da compra.", "cobranca", "negacao", ""),
    ("Não autorizei essa renovação automática e quero o dinheiro de volta.", "cobranca", "negacao", ""),
    ("Ainda não caiu o estorno do pagamento duplicado.", "cobranca", "negacao", ""),
    ("Não consigo entender por que o valor da mensalidade subiu sem aviso.", "cobranca", "negacao", ""),
    ("Não preciso de suporte técnico, preciso de um boleto atualizado.", "cobranca", "negacao", ""),
    ("Não encerrem minha conta: só quero trocar a forma de pagamento para boleto.", "cobranca", "negacao", ""),

    ("Fui cobrado depois de pedir o cancelamento, quero o valor de volta.", "cobranca", "ambiguo", "cancelamento"),
    ("O pagamento foi debitado, mas o app mostrou erro e o pedido não foi confirmado.", "cobranca", "ambiguo", "tecnico"),
    ("Paguei pelo frete expresso e o pedido chegou no prazo normal; quero a diferença.", "cobranca", "ambiguo", "entrega"),
    ("Minha conta foi bloqueada por falta de pagamento, mas a fatura está paga.", "cobranca", "ambiguo", "acesso"),
    ("Cobraram a assinatura de um serviço que eu já tinha encerrado.", "cobranca", "ambiguo", "cancelamento"),
    ("O produto veio com defeito e quero o reembolso, não a troca.", "cobranca", "ambiguo", "entrega"),
    ("Se não estornarem a cobrança duplicada até sexta, vou cancelar.", "cobranca", "ambiguo", "cancelamento"),
    ("Recebi um e-mail de cobrança de uma conta que não é minha.", "cobranca", "ambiguo", "acesso"),

    # ---------------- acesso ----------------
    ("Esqueci minha senha e o e-mail de recuperação não chega.", "acesso", "direto", ""),
    ("Minha conta foi bloqueada após três tentativas de login.", "acesso", "direto", ""),
    ("O código de verificação por SMS não chega no meu celular.", "acesso", "direto", ""),
    ("Troquei de número e agora não consigo receber o código de dois fatores.", "acesso", "direto", ""),
    ("Quero alterar o e-mail de login da minha conta.", "acesso", "direto", ""),
    ("Aparece \"usuário ou senha inválidos\", mas tenho certeza de que a senha está certa.", "acesso", "direto", ""),
    ("Acho que alguém entrou na minha conta, recebi alerta de acesso de outro estado.", "acesso", "direto", ""),
    ("Como adiciono outro usuário com acesso à conta da empresa?", "acesso", "direto", ""),
    ("O link para redefinir a senha diz que expirou.", "acesso", "direto", ""),
    ("Não lembro qual e-mail usei para criar a conta.", "acesso", "direto", ""),
    ("Preciso remover o acesso de um funcionário que saiu da empresa.", "acesso", "direto", ""),
    ("O login com Google parou de funcionar para minha conta.", "acesso", "direto", ""),

    ("Não quero cancelar minha conta, só não consigo entrar nela desde ontem.", "acesso", "negacao", ""),
    ("Não fui cobrada errado, o problema é que perdi o acesso à conta.", "acesso", "negacao", ""),
    ("O app não está travando; ele abre normalmente, mas não aceita minha senha.", "acesso", "negacao", ""),
    ("Não recebo o código de verificação de jeito nenhum.", "acesso", "negacao", ""),
    ("Ninguém da minha equipe consegue fazer login com as credenciais novas.", "acesso", "negacao", ""),
    ("Não é a entrega que me preocupa: alguém trocou o e-mail da minha conta sem eu pedir.", "acesso", "negacao", ""),
    ("Não reconheço esse login feito às 3h da manhã na minha conta.", "acesso", "negacao", ""),
    ("Não consigo redefinir a senha porque não tenho mais acesso ao e-mail antigo.", "acesso", "negacao", ""),
    ("Não tenho o celular cadastrado comigo, como entro sem o segundo fator?", "acesso", "negacao", ""),
    ("Não precisa estornar nada; só desbloqueiem meu usuário, por favor.", "acesso", "negacao", ""),

    ("Depois da atualização, o app pede login toda hora e não salva a sessão.", "acesso", "ambiguo", "tecnico"),
    ("Quero encerrar o acesso do meu ex-sócio, mas manter a conta ativa.", "acesso", "ambiguo", "cancelamento"),
    ("Minha conta foi suspensa e não sei se é por pagamento ou por segurança.", "acesso", "ambiguo", "cobranca"),
    ("Não consigo entrar para acompanhar meu pedido.", "acesso", "ambiguo", "entrega"),
    ("Ao tentar entrar, dá erro de autenticação e volta para a tela inicial.", "acesso", "ambiguo", "tecnico"),
    ("Recebi um e-mail pedindo para confirmar minha senha; é golpe?", "acesso", "ambiguo", "tecnico"),
    ("Quero saber quem tem acesso às faturas da empresa.", "acesso", "ambiguo", "cobranca"),
    ("Perdi o acesso e por isso não consegui pagar a fatura em dia.", "acesso", "ambiguo", "cobranca"),

    # ---------------- tecnico ----------------
    ("O aplicativo fecha sozinho toda vez que abro a tela de configurações.", "tecnico", "direto", ""),
    ("O site retorna erro 500 quando tento gerar o relatório.", "tecnico", "direto", ""),
    ("A exportação para Excel trava em 90% e não termina.", "tecnico", "direto", ""),
    ("As notificações pararam de chegar depois da última atualização.", "tecnico", "direto", ""),
    ("O sistema está muito lento desde hoje de manhã.", "tecnico", "direto", ""),
    ("A integração com a API devolve timeout em todas as chamadas.", "tecnico", "direto", ""),
    ("Os gráficos do painel aparecem em branco no Chrome.", "tecnico", "direto", ""),
    ("O upload de arquivos PDF falha com a mensagem \"formato não suportado\".", "tecnico", "direto", ""),
    ("A sincronização entre o celular e o computador parou de funcionar.", "tecnico", "direto", ""),
    ("O botão de salvar não responde no formulário de cadastro.", "tecnico", "direto", ""),
    ("Depois que atualizei o Android, o app não abre mais.", "tecnico", "direto", ""),
    ("O relatório mensal está mostrando dados duplicados.", "tecnico", "direto", ""),

    ("Não quero cancelar nada, só preciso que o app pare de fechar sozinho.", "tecnico", "negacao", ""),
    ("Não é problema de senha: eu entro normalmente, mas a tela fica em branco depois.", "tecnico", "negacao", ""),
    ("Minha fatura está correta; o que não funciona é a geração do relatório.", "tecnico", "negacao", ""),
    ("O pedido não tem nada a ver com isso, o problema é o site que não carrega.", "tecnico", "negacao", ""),
    ("A exportação não funciona nem no Chrome nem no Firefox.", "tecnico", "negacao", ""),
    ("Não consigo anexar arquivos maiores que 2 MB, sempre dá erro.", "tecnico", "negacao", ""),
    ("Não aparece nenhum dado no painel desde a atualização.", "tecnico", "negacao", ""),
    ("Nenhuma das notificações push está chegando.", "tecnico", "negacao", ""),
    ("Não fui cobrado errado; o app é que mostra o saldo zerado por erro.", "tecnico", "negacao", ""),
    ("Não estou pedindo reembolso, quero que corrijam o erro na emissão do relatório.", "tecnico", "negacao", ""),

    ("Tentei pagar, mas o botão de confirmar não funciona.", "tecnico", "ambiguo", "cobranca"),
    ("O rastreio do pedido mostra erro na página.", "tecnico", "ambiguo", "entrega"),
    ("O app trava tanto que estou pensando em cancelar.", "tecnico", "ambiguo", "cancelamento"),
    ("Depois de trocar a senha, o app começou a travar na abertura.", "tecnico", "ambiguo", "acesso"),
    ("Não consigo emitir a segunda via porque a página dá erro.", "tecnico", "ambiguo", "cobranca"),
    ("O site fica fora do ar e não consigo acompanhar a entrega.", "tecnico", "ambiguo", "entrega"),
    ("A tela de cadastro de novos usuários dá erro ao salvar.", "tecnico", "ambiguo", "acesso"),
    ("O sistema está lento e o suporte disse que é limitação do meu plano.", "tecnico", "ambiguo", "cobranca"),

    # ---------------- entrega ----------------
    ("Meu pedido está parado na transportadora há uma semana.", "entrega", "direto", ""),
    ("O código de rastreio não mostra nenhuma atualização.", "entrega", "direto", ""),
    ("O produto chegou quebrado.", "entrega", "direto", ""),
    ("Recebi um item diferente do que comprei.", "entrega", "direto", ""),
    ("A entrega estava prevista para ontem e não chegou.", "entrega", "direto", ""),
    ("Quero trocar o endereço de entrega do pedido 4521.", "entrega", "direto", ""),
    ("Faltou um item no pacote que recebi.", "entrega", "direto", ""),
    ("O entregador marcou como entregue, mas não recebi nada.", "entrega", "direto", ""),
    ("Quero trocar o tamanho da camiseta que chegou.", "entrega", "direto", ""),
    ("Qual é o prazo de entrega para Manaus?", "entrega", "direto", ""),
    ("A caixa chegou aberta e com sinais de violação.", "entrega", "direto", ""),
    ("Preciso agendar a entrega para depois das 18h.", "entrega", "direto", ""),

    ("Não quero cancelar a compra, só quero saber quando o pedido vai chegar.", "entrega", "negacao", ""),
    ("Não preciso de reembolso, quero que enviem o item que faltou.", "entrega", "negacao", ""),
    ("O pagamento já foi aprovado; o problema é que o pedido não saiu do centro de distribuição.", "entrega", "negacao", ""),
    ("Ainda não recebi o pedido feito há dez dias.", "entrega", "negacao", ""),
    ("O pedido não chegou e o rastreio não atualiza.", "entrega", "negacao", ""),
    ("Não encontro a encomenda e o porteiro diz que não recebeu nada.", "entrega", "negacao", ""),
    ("Não é erro no app: o pedido realmente consta como devolvido ao remetente.", "entrega", "negacao", ""),
    ("Nenhum dos três volumes do pedido foi entregue.", "entrega", "negacao", ""),
    ("Não era esse o modelo que eu pedi.", "entrega", "negacao", ""),
    ("Não tenho problema com a conta, só quero mudar o endereço de entrega.", "entrega", "negacao", ""),

    ("O produto chegou com defeito; quero trocar por outro igual.", "entrega", "ambiguo", "cobranca"),
    ("O pedido atrasou tanto que já não sei se ainda quero.", "entrega", "ambiguo", "cancelamento"),
    ("Paguei o frete, mas o pedido ainda não foi enviado.", "entrega", "ambiguo", "cobranca"),
    ("Não consigo ver o status do pedido no app.", "entrega", "ambiguo", "tecnico"),
    ("O pedido chegou incompleto e o valor cobrado foi do pedido inteiro.", "entrega", "ambiguo", "cobranca"),
    ("Mudei de endereço e não sei se o pedido vai para o antigo.", "entrega", "ambiguo", "sem_informacao"),
    ("A transportadora diz que o endereço está errado, mas ele confere com meu cadastro.", "entrega", "ambiguo", "tecnico"),
    ("Recebi duas unidades e só comprei uma.", "entrega", "ambiguo", "cobranca"),

    # ---------------- cancelamento ----------------
    ("Quero cancelar minha assinatura a partir do próximo mês.", "cancelamento", "direto", ""),
    ("Por favor, encerrem minha conta e apaguem meus dados.", "cancelamento", "direto", ""),
    ("Como faço para cancelar o plano anual?", "cancelamento", "direto", ""),
    ("Desejo cancelar o pedido 7788 antes do envio.", "cancelamento", "direto", ""),
    ("Quero desistir da compra dentro do prazo de sete dias.", "cancelamento", "direto", ""),
    ("Cancelem a renovação automática do meu plano.", "cancelamento", "direto", ""),
    ("Vou migrar para outro fornecedor e quero cancelar o contrato.", "cancelamento", "direto", ""),
    ("Solicito o cancelamento da linha adicional.", "cancelamento", "direto", ""),
    ("Quero cancelar o serviço de backup contratado no mês passado.", "cancelamento", "direto", ""),
    ("Gostaria de encerrar a conta da empresa, que foi fechada.", "cancelamento", "direto", ""),
    ("Peço o cancelamento imediato do contrato, conforme a cláusula de fidelidade.", "cancelamento", "direto", ""),
    ("Quero cancelar a assinatura premium.", "cancelamento", "direto", ""),

    ("Não quero mais continuar com o plano; cancelem, por favor.", "cancelamento", "negacao", ""),
    ("Não tenho mais interesse no serviço e quero encerrar a assinatura.", "cancelamento", "negacao", ""),
    ("Não é problema técnico nem de cobrança: eu simplesmente quero cancelar.", "cancelamento", "negacao", ""),
    ("Não quero trocar o produto, quero cancelar o pedido.", "cancelamento", "negacao", ""),
    ("Não renovem meu plano; quero encerrar no fim do ciclo.", "cancelamento", "negacao", ""),
    ("Não preciso de desconto para ficar, quero cancelar mesmo.", "cancelamento", "negacao", ""),
    ("Não tenho tempo para usar o serviço, então prefiro cancelar.", "cancelamento", "negacao", ""),
    ("Não quero reembolso parcial, quero cancelar o contrato inteiro.", "cancelamento", "negacao", ""),
    ("Nunca mais vou usar, encerrem minha conta.", "cancelamento", "negacao", ""),
    ("Não quero suporte técnico, quero o cancelamento da assinatura.", "cancelamento", "negacao", ""),

    ("Quero cancelar e receber de volta o valor do mês.", "cancelamento", "ambiguo", "cobranca"),
    ("Quero cancelar porque o app vive travando.", "cancelamento", "ambiguo", "tecnico"),
    ("Quero cancelar o pedido porque está atrasado demais.", "cancelamento", "ambiguo", "entrega"),
    ("Quero reduzir meu plano para o básico ou cancelar, o que for mais barato.", "cancelamento", "ambiguo", "cobranca"),
    ("Encerrem minha conta, pois não consigo mais acessá-la mesmo.", "cancelamento", "ambiguo", "acesso"),
    ("Quero cancelar a compra e devolver o produto que chegou errado.", "cancelamento", "ambiguo", "entrega"),
    ("Desisti: cancelem tudo e estornem o que foi cobrado.", "cancelamento", "ambiguo", "cobranca"),
    ("Quero cancelar o plano empresarial e manter só o individual.", "cancelamento", "ambiguo", "cobranca"),

    # ---------------- sem_informacao ----------------
    ("Oi, preciso de ajuda.", "sem_informacao", "insuficiente", ""),
    ("Não está funcionando.", "sem_informacao", "insuficiente", ""),
    ("Bom dia, tenho um problema.", "sem_informacao", "insuficiente", ""),
    ("Isso de novo?!", "sem_informacao", "insuficiente", ""),
    ("Vocês podem me ligar?", "sem_informacao", "insuficiente", ""),
    ("Já falei com vocês ontem sobre aquilo.", "sem_informacao", "insuficiente", ""),
    ("Continua igual.", "sem_informacao", "insuficiente", ""),
    ("Urgente!!!", "sem_informacao", "insuficiente", ""),
    ("Não sei explicar, mas algo está errado.", "sem_informacao", "insuficiente", ""),
    ("Quero falar com um atendente.", "sem_informacao", "insuficiente", ""),
    ("O protocolo 3381 ainda não foi resolvido.", "sem_informacao", "insuficiente", ""),
    ("Segue o print em anexo.", "sem_informacao", "insuficiente", ""),
    ("???", "sem_informacao", "insuficiente", ""),
    ("Por favor, verifiquem minha situação.", "sem_informacao", "insuficiente", ""),
    ("Não deu certo.", "sem_informacao", "insuficiente", ""),
    ("Oi", "sem_informacao", "insuficiente", ""),
    ("Alguém pode me responder?", "sem_informacao", "insuficiente", ""),
    ("Estou esperando há dias.", "sem_informacao", "insuficiente", ""),
    ("Mesma coisa do chamado anterior.", "sem_informacao", "insuficiente", ""),
    ("Tem como resolver isso hoje?", "sem_informacao", "insuficiente", ""),
    ("Preciso de uma posição.", "sem_informacao", "insuficiente", ""),
    ("Não entendi o que aconteceu.", "sem_informacao", "insuficiente", ""),
    ("Teste", "sem_informacao", "insuficiente", ""),
    ("Vocês resolveram?", "sem_informacao", "insuficiente", ""),
    ("Quero fazer uma reclamação.", "sem_informacao", "insuficiente", ""),
    ("Aconteceu de novo e não é a primeira vez.", "sem_informacao", "insuficiente", ""),
    ("Olá, poderiam verificar?", "sem_informacao", "insuficiente", ""),
    ("Ainda não.", "sem_informacao", "insuficiente", ""),
    ("Estou muito insatisfeita.", "sem_informacao", "insuficiente", ""),
    ("Me liga quando puder.", "sem_informacao", "insuficiente", ""),
]

NEGACAO = re.compile(r"\b(nao|nunca|nenhum|nenhuma|nem|ninguem|nada|jamais)\b")


def sem_acento(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", texto.lower()) if unicodedata.category(c) != "Mn")


def particionar(linhas):
    rng = random.Random(SEMENTE)
    estratos = {}
    for i, linha in enumerate(linhas):
        estratos.setdefault((linha["categoria"], linha["tipo"]), []).append(i)
    particao = {}
    for chave in sorted(estratos):
        idx = estratos[chave][:]
        rng.shuffle(idx)
        n_ajuste = round(FRACAO_AJUSTE * len(idx))
        for j, i in enumerate(idx):
            particao[i] = "ajuste" if j < n_ajuste else "avaliacao"
    return particao


def main():
    textos = [t for t, *_ in CHAMADOS]
    assert len(textos) == len(set(textos)), "texto duplicado"
    linhas = [
        {"id": f"CH{i:03d}", "texto": t, "categoria": c, "tipo": tp, "rotulo_alternativo": alt,
         "contem_negacao": int(bool(NEGACAO.search(sem_acento(t))))}
        for i, (t, c, tp, alt) in enumerate(CHAMADOS, 1)
    ]
    part = particionar(linhas)
    for i, linha in enumerate(linhas):
        linha["particao"] = part[i]
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    with SAIDA.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)
    print(f"{len(linhas)} chamados -> {SAIDA}")


if __name__ == "__main__":
    main()
