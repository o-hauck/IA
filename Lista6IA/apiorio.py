import pandas as pd
from apyori import apriori

# Carregar os dados
base = pd.read_csv('MercadoSim.csv', sep=';', encoding='cp1252', header=None)

# Pegar todos os itens únicos presentes na base
itens_unicos = pd.unique(base.values.ravel())
itens_unicos = [item for item in itens_unicos if pd.notna(item)]

# Montar transações com presença/ausência
transacoes = []
for i in range(base.shape[0]):
    linha = base.iloc[i, :].dropna().tolist()
    transacao = []

    for item in itens_unicos:
        if item in linha:
            transacao.append(f'{item}=Presente')
        else:
            transacao.append(f'{item}=Ausente')

    transacoes.append(transacao)

# Executar Apriori
regras = apriori(transacoes, min_support=0.3, min_confidence=0.6)
saida = list(regras)

# Extrair regras para DataFrame
Antecedente = []
Consequente = []
suporte = []
confianca = []
lift = []

for resultado in saida:
    s = resultado.support
    for regra in resultado.ordered_statistics:
        a = list(regra.items_base)
        b = list(regra.items_add)
        c = regra.confidence
        l = regra.lift
        if len(a) == 0 or len(b) == 0:
            continue
        Antecedente.append(a)
        Consequente.append(b)
        suporte.append(s)
        confianca.append(c)
        lift.append(l)

# Montar DataFrame das regras
df_regras = pd.DataFrame({
    'Antecedente': Antecedente,
    'Consequente': Consequente,
    'suporte': suporte,
    'confianca': confianca,
    'lift': lift
})

# Ordenar por lift (interesse)
df_regras = df_regras.sort_values(by='lift', ascending=False)

# Exibir apenas as regras
print("\nRegras de associação (incluindo ausência de itens):\n")
for index, row in df_regras.iterrows():
    antecedente = ', '.join(row['Antecedente'])
    consequente = ', '.join(row['Consequente'])
    print(f"Se [{antecedente}] então [{consequente}] (suporte={row['suporte']:.2f}, confiança={row['confianca']:.2f}, lift={row['lift']:.2f})")
