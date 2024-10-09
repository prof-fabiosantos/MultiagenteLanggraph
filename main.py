# Importa a função load_dotenv para carregar variáveis de ambiente a partir de um arquivo .env
from dotenv import load_dotenv

# Importa o módulo os para acessar variáveis de ambiente e interagir com o sistema operacional
import os

# Importa a função create_graph, que será utilizada para criar o grafo da aplicação (presumivelmente definido no arquivo 'graph')
from graph import create_graph

# Importa tipos os dicionários tipados do módulo typing para facilitar a tipagem explícita de funções e classes
from typing import TypedDict

# Importa a classe StateGraph e a constante END da biblioteca langgraph.graph para criação e manipulação de um grafo de estado
from langgraph.graph import StateGraph, END


# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Obtém o valor da chave da variável de ambiente "OPENAI_API_KEY"
openai_api_key = os.getenv("OPENAI_API_KEY")

# Verifica se a chave foi carregada corretamente
if openai_api_key:
    # Se a chave foi encontrada, define-a como uma variável de ambiente
    os.environ["OPENAI_API_KEY"] = openai_api_key
else:
    # Caso contrário, lança um erro informando que a chave não foi encontrada
    raise ValueError("A chave OPENAI_API_KEY não foi encontrada no arquivo .env")


# Define um dicionário tipado que espera uma string de entrada e um booleano
class UserInput(TypedDict):
    input: str
    continue_conversation: bool

# Função para obter a entrada do usuário e decidir se a conversa continua
def get_user_input(state: UserInput) -> UserInput:
    # Solicita ao usuário que digite uma questão
    user_input = input("\nDigite sua questão (ou 'q' para sair) : ")
    # Retorna um dicionário com a entrada do usuário e uma flag para continuar a conversa
    return {
        "input": user_input,
        "continue_conversation": user_input.lower() != 'q'  # Verifica se o usuário quer sair
    }

# Função que processa a questão do usuário
def process_question(state: UserInput):
    # Cria um grafo de processamento de questões
    graph = create_graph()        
    # Invoca o grafo com a entrada do usuário
    result = graph.invoke({"input": state["input"]})
    # Exibe a resposta final
    print("\n--- Final answer ---")
    print(result["output"])
    # Retorna o estado atual
    return state

# Função que cria o grafo da conversa
def create_conversation_graph():
    # Inicializa um grafo de estado com o tipo de entrada UserInput
    workflow = StateGraph(UserInput)

    # Adiciona o nó "get_input" ao grafo, associando-o à função get_user_input
    workflow.add_node("get_input", get_user_input)
    # Adiciona o nó "process_question" ao grafo, associando-o à função process_question
    workflow.add_node("process_question", process_question)

    # Define o ponto de entrada do grafo como o nó "get_input"
    workflow.set_entry_point("get_input")

    # Adiciona arestas condicionais ao nó "get_input", direcionando a lógica de fluxo
    workflow.add_conditional_edges(
        "get_input",
        # Função que decide se a conversa continua ou termina
        lambda x: "continue" if x["continue_conversation"] else "end",
        {
            "continue": "process_question",  # Continua para "process_question"
            "end": END  # Termina a conversa
        }
    )

    # Adiciona uma aresta do nó "process_question" de volta para "get_input"
    workflow.add_edge("process_question", "get_input")

    # Compila e retorna o grafo
    return workflow.compile()

# Função principal do programa
def main():
    # Cria o grafo da conversa
    conversation_graph = create_conversation_graph()
    # Inicia a conversa invocando o grafo com o estado inicial
    conversation_graph.invoke({"input": "", "continue_conversation": True})

# Ponto de entrada do script
if __name__ == "__main__":
    # Executa a função main se o script for executado diretamente
    main()
