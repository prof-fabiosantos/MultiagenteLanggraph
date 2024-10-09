# Importa a classe StateGraph e a constante END do módulo langgraph.graph para criar e gerenciar um grafo de estado
from langgraph.graph import StateGraph, END

# Importa o módulo para facilitar a tipagem explícita e dicionários tipados
from typing import TypedDict

# Importa três funções (ou agentes) que serão utilizadas para analisar e responder perguntas específicas e genéricas
from agents import analyze_question, answer_legislation_question, answer_generic_question

# Define uma classe tipada chamada AgentState, que descreve o estado de entrada, saída e decisão no grafo
class AgentState(TypedDict):
    input: str  # A entrada fornecida ao agente (ex: a pergunta do usuário)
    output: str  # A saída gerada após o processamento da entrada
    decision: str  # A decisão tomada após a análise, usada para determinar o próximo nó no grafo

# Função que cria e retorna o grafo de três passos que processa a entrada de acordo com a decisão
def create_graph():
    # Inicializa um grafo de estado que usa AgentState como estrutura de dados para os nós
    workflow = StateGraph(AgentState)

    # Adiciona o nó "analyze" ao grafo, associando-o à função analyze_question
    workflow.add_node("analyze", analyze_question)

    # Adiciona o nó "legislation_agent", associando-o à função answer_legislation_question
    workflow.add_node("legislation_agent", answer_legislation_question)

    # Adiciona o nó "generic_agent", associando-o à função answer_generic_question
    workflow.add_node("generic_agent", answer_generic_question)

    # Define arestas condicionais para o nó "analyze", direcionando a execução com base na decisão retornada
    workflow.add_conditional_edges(
        "analyze",  # Nó de origem
        lambda x: x["decision"],  # Função que determina para onde ir, baseada na chave "decision"
        {
            # Se a decisão for "legislation", o fluxo vai para o nó "legislation_agent"
            "legislation": "legislation_agent",
            # Se a decisão for "general", o fluxo vai para o nó "generic_agent"
            "general": "generic_agent"
        }
    )

    # Define "analyze" como o ponto de entrada (primeiro nó) do grafo
    workflow.set_entry_point("analyze")

    # Adiciona uma aresta do nó "legislation_agent" para o nó final END, indicando que esse é o fim do fluxo
    workflow.add_edge("legislation_agent", END)

    # Adiciona uma aresta do nó "generic_agent" para o nó final END, também indicando o fim do fluxo
    workflow.add_edge("generic_agent", END)

    # Compila e retorna o grafo para ser utilizado posteriormente
    return workflow.compile()
