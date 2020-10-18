#===============================================================================
# TRABALHO 01 - classificador para apoio à decisão de aprovação de crédito.
#===============================================================================

#-------------------------------------------------------------------------------
# Importar bibliotecas
#-------------------------------------------------------------------------------

import pandas as pd
import numpy as np

from sklearn.neighbors     import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score


from matplotlib import pyplot as plt

#-------------------------------------------------------------------------------
# Ler o arquivo CSV com os dados do conjunto de treinamento
#-------------------------------------------------------------------------------

dados = pd.read_csv('conjunto_de_treinamento.csv')  
#dados_teste = pd.read_csv('conjunto_de_teste.csv')

        
#-------------------------------------------------------------------------------
# Explorar os dados
#-------------------------------------------------------------------------------

print ( '\nImprimir o conjunto de dados:\n')

print(dados)

print ( '\nImprimir o conjunto de dados transposto')
print ('para visualizar os nomes de todas as colunas:\n')

print(dados.T)

print ( '\nImprimir os tipos de cada variável:\n')

print(dados.dtypes)

print ( '\nIdentificar as variáveis categóricas:\n')


#todas as variaveis object ou não quantitativox

variaveis_categoricas = [
    x for x in dados.columns if dados[x].dtype == 'object' #or x == 'inadimplente'
    ]

print(variaveis_categoricas)


print ( '\nVerificar a cardinalidade de cada variável categórica:')
print ( 'obs: cardinalidade = qtde de valores distintos que a variável pode assumir\n')

for v in variaveis_categoricas:
    
    print ('\n%15s:'%v , "%4d categorias" % len(dados[v].unique()))
    print (dados[v].unique(),'\n')    



#-------------------------------------------------------------------------------
# Executar preprocessamento dos dados
#-------------------------------------------------------------------------------

'''id_solicitante                   ----> não-ordinal 40k categorias --> Descartar=ok           
produto_solicitado                  ---->             
dia_vencimento                      ---->
forma_envio_solicitacao             ----> não-ordinal   4 categorias --> Descartar=ok    
tipo_endereco                       ---->            
sexo                                ----> não-ordinal   4 categorias --> One-Hot-Encoding=ok           
idade                               ---->            
estado_civil                        ---->             
qtde_dependentes                    ---->             
grau_instrucao                      ---->             
nacionalidade                       ---->         
estado_onde_nasceu                  ----> não-ordinal  28 categorias --> Descartar=ok     
estado_onde_reside                  ----> não-ordinal  27 categorias --> Descartar=ok     
possui_telefone_residencial         ----> binária       2 categorias -->     
codigo_area_telefone_residencial    ----> não-ordinal  95 categorias --> Descartar=ok                
tipo_residencia                     ---->     
meses_na_residencia                 ---->       
possui_telefone_celular             ----> não-ordinal   1 categorias -->   
possui_email                        ---->     
renda_mensal_regular                ---->   
renda_extra                         ---->    
possui_cartao_visa                  ---->     
possui_cartao_mastercard            ---->     
possui_cartao_diners                ---->                       
possui_cartao_amex                  ---->     
possui_outros_cartoes               ---->     
qtde_contas_bancarias               ---->    
qtde_contas_bancarias_especiais     ---->     
valor_patrimonio_pessoal            ---->     
possui_carro                        ---->     
vinculo_formal_com_empresa          ----> binária      2 categorias -->   
estado_onde_trabalha                ----> não-ordinal 28 categorias -->                                  
possui_telefone_trabalho            ----> binária      2 categorias -->     
codigo_area_telefone_trabalho       ----> não-ordinal 84 categorias --> Descartar=ok                            
meses_no_trabalho                   ---->     
profissao                           ---->     
ocupacao                            ---->     
profissao_companheiro               ---->   
local_onde_reside                   ---->   
local_onde_trabalha                 ---->   
inadimplente                        ---->'''  





print (dados.T)
dados = dados.drop(['id_solicitante',
                    'forma_envio_solicitacao',
                    'estado_onde_trabalha',
                    'estado_onde_nasceu',
                    'estado_onde_reside',
                    'codigo_area_telefone_residencial',
                    'codigo_area_telefone_trabalho'
                    ],axis=1)
print (dados.T)


#-------------------------------------------------------------------------------
# TRATAR DADOS FALTANTES
#
#Substituir os dados pela média ou mediana(verificar melhor opção) 
#-------------------------------------------------------------------------------


#dados2 = dados.dropna() #ELIMINA LINHA DE VALORES NULOS

#enulo = dados.isnull()  #VERIFICA QUAIS SÃO NULOS

#faltante = dados.isnull().sum() #LISTA COLUNAS NULAS

#faltantes_percentual = (dados.isnull().sum() / len(dados['id_solicitante']))*100 #percentual dos Dados Faltantes

dados['tipo_residencia'].fillna(dados['tipo_residencia'].mean(),inplace = True)
dados['meses_na_residencia'].fillna(dados['meses_na_residencia'].mean(),inplace = True)
dados['profissao'].fillna(dados['profissao'].mean(),inplace = True)
dados['ocupacao'].fillna(dados['ocupacao'].mean(),inplace = True)
dados['profissao_companheiro'].fillna(dados['profissao_companheiro'].mean(),inplace = True)
dados['grau_instrucao_companheiro'].fillna(dados['grau_instrucao_companheiro'].mean(),inplace = True)
dados['local_onde_reside'].fillna(dados['local_onde_reside'].mean(),inplace = True)
dados['local_onde_trabalha'].fillna(dados['local_onde_trabalha'].mean(),inplace = True)


print (dados.T)


print ( '\nAplicar one-hot encoding nas variáveis que tenham')
print ( '3 ou mais categorias:')

dados = pd.get_dummies(dados,columns=['sexo'])
print (dados.head(5).T)

print ( '\nAplicar binarização simples nas variáveis que tenham')
print ( 'apenas 2 categorias:\n') 
     
binarizador = LabelBinarizer()
for v in ['possui_telefone_residencial',
          'vinculo_formal_com_empresa',
          'possui_telefone_trabalho',
          'possui_telefone_celular']:
    dados[v] = binarizador.fit_transform(dados[v])
print (dados.head(5).T)

print ( '\nVerificar a quantidade de amostras de cada classe:\n')

print(dados['inadimplente'].value_counts())

print ( '\nVerificar o valor médio de cada atributo em cada classe:')

print(dados.groupby(['inadimplente']).mean().T)

#-------------------------------------------------------------------------------
# Plotar diagrama de dispersão por classe
#-------------------------------------------------------------------------------
  
atributo1 = 'local_onde_trabalha'
atributo2 = 'idade'

cores = [ 'red' if x else 'blue' for x in dados['inadimplente'] ]

grafico = dados.plot.scatter(
    atributo1,
    atributo2,
    c      = cores,
    s      = 10,
    marker = 'o',
    alpha  = 0.5,
    figsize = (14,14)
    )

plt.show()

#-------------------------------------------------------------------------------
# Selecionar os atributos que serão utilizados pelo classificador
#-------------------------------------------------------------------------------

atributos_selecionados = [
    'produto_solicitado',
    'dia_vencimento',
    'tipo_endereco',
    'idade',                       
    'estado_civil',
    'qtde_dependentes',
    #'grau_instrucao',
    'nacionalidade',
    'possui_telefone_residencial',
    'tipo_residencia',
    'meses_na_residencia',
    #'possui_telefone_celular',
    'possui_email',
    'renda_mensal_regular',
    'renda_extra',
    'possui_cartao_visa',
    'possui_cartao_mastercard',
    'possui_cartao_diners',
    #'possui_outros_cartoes',
    'qtde_contas_bancarias',
    'qtde_contas_bancarias_especiais',
    'valor_patrimonio_pessoal',
    'possui_carro',
    'vinculo_formal_com_empresa',
    'possui_telefone_trabalho',
    'meses_no_trabalho',
    'profissao',
    'ocupacao', 
    'profissao_companheiro',
    'grau_instrucao_companheiro',
    'local_onde_reside',
    'local_onde_trabalha',                         
    #'sexo_',
    'sexo_F',
    'sexo_M',
     #'sexo_N'
    'inadimplente'
    ]

dados = dados[atributos_selecionados]
#alvo = dados['inadimplente'].astype(bool)
#-------------------------------------------------------------------------------
# Embaralhar o conjunto de dados para garantir que a divisão entre os dados de
# treino e os dados de teste esteja isenta de qualquer viés de seleção
#-------------------------------------------------------------------------------

dados_embaralhados = dados.sample(frac=1,random_state=12345)


#-------------------------------------------------------------------------------
# Criar os arrays X e Y separando atributos e alvo
#-------------------------------------------------------------------------------

x = dados_embaralhados.loc[:,dados_embaralhados.columns!='inadimplente'].values
#y = alvo.sample(frac=1,random_state=12345)
y = dados_embaralhados.loc[:,dados_embaralhados.columns=='inadimplente'].values



#-------------------------------------------------------------------------------
# Separar X e Y em conjunto de treino e conjunto de teste
#-------------------------------------------------------------------------------


#q = 30000  # qtde de amostras selecionadas para treinamento

# conjunto de treino

#x_treino = x[:q,:]
#y_treino = y[:q].ravel()

# conjunto de teste

#x_teste = x[q:,:]
#y_teste = y[q:].ravel()

x_treino, x_teste, y_treino, y_teste = train_test_split(
    x, 
    y.ravel(),
    train_size=30000,
    shuffle=True,
    random_state=777
    )


#-------------------------------------------------------------------------------
# Ajustar a escala dos atributos nos conjuntos de treino e de teste
#-------------------------------------------------------------------------------

ajustador_de_escala = MinMaxScaler()
ajustador_de_escala.fit(x_treino)

x_treino = ajustador_de_escala.transform(x_treino)
x_teste  = ajustador_de_escala.transform(x_teste)

#-------------------------------------------------------------------------------
# Treinar um classificador KNN com o conjunto de treino
#-------------------------------------------------------------------------------

classificador = KNeighborsClassifier(n_neighbors=20)

classificador = classificador.fit(x_treino,y_treino)


#-------------------------------------------------------------------------------
# Obter as respostas do classificador no mesmo conjunto onde foi treinado
#-------------------------------------------------------------------------------

y_resposta_treino = classificador.predict(x_treino)

#-------------------------------------------------------------------------------
# Obter as respostas do classificador no conjunto de teste
#-------------------------------------------------------------------------------

y_resposta_teste = classificador.predict(x_teste)

#-------------------------------------------------------------------------------
# Verificar a acurácia do classificador
#-------------------------------------------------------------------------------

print ("\nDESEMPENHO DENTRO DA AMOSTRA DE TREINO\n")
    
total   = len(y_treino)
acertos = sum(y_resposta_treino==y_treino)
erros   = sum(y_resposta_treino!=y_treino)

print ("Total de amostras: " , total)
print ("Respostas corretas:" , acertos)
print ("Respostas erradas: " , erros)

acuracia = acertos / total

print ("Acurácia = %.1f %%" % (100*acuracia))

print ("\nDESEMPENHO FORA DA AMOSTRA DE TREINO\n")

total   = len(y_teste)
acertos = sum(y_resposta_teste==y_teste)
erros   = sum(y_resposta_teste!=y_teste)

print ("Total de amostras: " , total)
print ("Respostas corretas:" , acertos)
print ("Respostas erradas: " , erros)

acuracia = acertos / total

print ("Acurácia = %.1f %%" % (100*acuracia))

#-------------------------------------------------------------------------------
# Verificar a variação da acurácia com o número de vizinhos
#-------------------------------------------------------------------------------

print ( "\n  K TREINO  TESTE")
print ( " -- ------ ------")

for k in range(1,26,2):

    classificador = KNeighborsClassifier(
        n_neighbors = k,
        weights     = 'distance',
        p           = 2
        )
    classificador = classificador.fit(x_treino,y_treino)

    y_resposta_treino = classificador.predict(x_treino)
    y_resposta_teste  = classificador.predict(x_teste)
    
    acuracia_treino = sum(y_resposta_treino==y_treino)/len(y_treino)
    acuracia_teste  = sum(y_resposta_teste ==y_teste) /len(y_teste)
    
    print(
        "%3d"%k,
        "%6.1f" % (100*acuracia_treino),
        "%6.1f" % (100*acuracia_teste)
        )


#-------------------------------------------------------------------------------
# Verificar a variação da acurácia com o número de vizinhos
# usando VALIDAÇÃO CRUZADA
#-------------------------------------------------------------------------------
    
for k in range(1,26,2):

    classificador = KNeighborsClassifier(
        n_neighbors = k,
        weights     = 'distances',
        p           = 2
        )

    scores = cross_val_score(
        classificador,
        x,
        y.ravel(), 
        cv=8
        )

    print (
        'k = %2d' % k,
        'scores =',scores,
        'acurácia média = %6.1f' % (100*sum(scores)/8)
        )


