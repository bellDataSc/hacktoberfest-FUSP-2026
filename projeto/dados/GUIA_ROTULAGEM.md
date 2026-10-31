# Guia de rotulagem e ficha do conjunto `chamados_ptbr.csv`

## O que é

180 chamados de atendimento **sintéticos**, em português do Brasil, escritos para este estudo. Nenhum texto vem de cliente real. Eles servem para testar a triagem em condições controladas, e não para estimar o desempenho em produção.

| coluna | significado |
|---|---|
| `id` | identificador estável (`CH001`…`CH180`) |
| `texto` | o chamado |
| `categoria` | rótulo de referência (regras abaixo) |
| `tipo` | `direto`, `negacao`, `ambiguo` ou `insuficiente` |
| `rotulo_alternativo` | segunda leitura plausível, preenchida só nos ambíguos |
| `contem_negacao` | 1 se o texto tem marcador léxico de negação (não, nunca, nenhum, nem, ninguém, nada, jamais) |
| `particao` | `ajuste` (72) ou `avaliacao` (108), estratificada por categoria × tipo, semente 2026 |

## Categorias

| chave | quando usar |
|---|---|
| `cobranca` | faturas, boletos, pagamentos, estornos, reembolsos, taxas, nota fiscal, forma de pagamento |
| `acesso` | login, senha, código de verificação, conta bloqueada, usuários e permissões, suspeita de invasão |
| `tecnico` | erros, falhas, lentidão, travamentos, telas em branco, integrações, relatórios com defeito |
| `entrega` | pedido, rastreio, atraso, produto danificado ou errado, troca, endereço de entrega |
| `cancelamento` | pedido explícito para cancelar ou encerrar assinatura, plano, conta, contrato ou pedido |
| `sem_informacao` | o texto não permite saber qual equipe deve atender |

## Regras de desempate

1. **A ação pedida vale mais que o contexto.** "Fui cobrado depois de cancelar, quero o valor de volta" pede um estorno → `cobranca`.
2. **Cancelamento só com pedido explícito.** Ameaça condicional ("se não resolverem, vou cancelar") ou intenção vaga ("estou pensando em cancelar") seguem a categoria do problema descrito.
3. **Um erro de sistema que impede a tarefa é `tecnico`**, a menos que o cliente peça uma ação de outra equipe (estorno, troca, desbloqueio).
4. **Uma menção negada não define a categoria.** "Não quero cancelar, só..." segue o que vem depois do "só".
5. Na dúvida entre uma categoria e `sem_informacao`, use `sem_informacao` apenas quando nenhuma equipe puder agir sem perguntar ao cliente.

## Tipos

- `direto`: uma única categoria clara.
- `negacao`: a negação é decisiva para o rótulo, seja porque nega uma categoria concorrente ("não é problema de senha"), seja porque é a forma do pedido ("não renovem meu plano").
- `ambiguo`: duas leituras plausíveis, resolvidas pelas regras acima. A outra leitura fica em `rotulo_alternativo`.
- `insuficiente`: sem informação para triagem.

## Limitações conhecidas

- Os textos são curtos e limpos, sem erros de digitação, gírias ou e-mails longos com histórico.
- O mesmo autor escreveu os textos e os rótulos. Uma segunda pessoa deve revisar os rótulos antes da avaliação final, e a concordância entre as duas deve ser registrada.
- Com 108 exemplos na avaliação, diferenças de poucos pontos percentuais ficam dentro do intervalo de confiança.
- As categorias e as regras de desempate foram definidas para este estudo. Outro desenho mudaria os resultados.
