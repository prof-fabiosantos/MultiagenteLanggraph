# Importa o modelo de chat da OpenAI da LangChain, utilizado para gerar respostas em linguagem natural
from langchain_openai import ChatOpenAI

# Importa a classe PromptTemplate da LangChain, que facilita a criação de templates de prompts
from langchain.prompts import PromptTemplate

# Importa a classe RetrievalQA, que permite construir cadeias de perguntas e respostas com recuperação de documentos
from langchain.chains import RetrievalQA

# Importa o banco de dados de vetores Chroma, que já foi criado previamente no arquivo vectorStore
from vectorStore import crhomadb

import warnings
# Ignorar o LangChainDeprecationWarning específico
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain.chains.base")

# Função para analisar a questão do usuário e decidir se é uma pergunta sobre legislação ou uma pergunta geral
def analyze_question(state):
    # Inicializa o modelo de chat da OpenAI
    llm = ChatOpenAI()
    
    # Cria um template de prompt que será usado para classificar a questão
    prompt = PromptTemplate.from_template("""
    You are an agent that needs to define if a question is a legislation one or a general one.

    Question : {input}

    Analyse the question. Only answer with "legislation" if the question is about legislation. If not just answer "general".

    Your answer (legislation/general) :
    """)
    
    # Concatena o template com o modelo de linguagem para formar uma cadeia de execução
    chain = prompt | llm
    
    # Executa a cadeia passando a entrada do usuário e obtém a resposta
    response = chain.invoke({"input": state["input"]})
    
    # Extrai a resposta do modelo, convertendo para minúsculas e removendo espaços
    decision = response.content.strip().lower()
    
    # Exibe o tipo de questão classificada (legislação ou geral)
    print("Tipo de questão: " + decision)
    
    # Retorna a decisão e a entrada original do usuário
    return {"decision": decision, "input": state["input"]}

# Função para responder a perguntas relacionadas à legislação
def answer_legislation_question(state):
    # Inicializa o modelo de chat da OpenAI
    llm = ChatOpenAI()
    
    # Cria um template de prompt para responder perguntas sobre legislação
    prompt_template = PromptTemplate.from_template(
        """
        You are an expert in legislation. Answer in portuguese the following question with step-by-step details:
        
        Question: {question}
        """
    )

    # Preenche o template com a pergunta do usuário
    formatted_prompt = prompt_template.format(question=state["input"])    

    # Configura o QA com recuperação de documentos (usando o banco de dados Chroma)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Define o tipo de cadeia a ser usada para as respostas
        retriever=crhomadb.as_retriever(),  # Usa o Chroma como retriever
        return_source_documents=True  # Configura para retornar os documentos relevantes junto com a resposta
    )

    # Executa a cadeia de perguntas e respostas com o prompt formatado
    #response = qa({"query": formatted_prompt})
    response = qa.invoke({"query": formatted_prompt})

    # Se houver documentos recuperados, exibe seu conteúdo
    if "source_documents" in response:
         print("Documentos recuperados:")
         for doc in response["source_documents"]:
             print(doc.page_content)

    # Retorna a resposta gerada pelo modelo de linguagem
    return {"output": response["result"]}

# Função para responder a perguntas gerais
def answer_generic_question(state):
    # Inicializa o modelo de chat da OpenAI
    llm = ChatOpenAI()
    
    # Cria um template de prompt para responder perguntas gerais de forma concisa
    prompt = PromptTemplate.from_template(
        "Give a general and concise answer to the question: {input}"
    )
    
    # Concatena o template com o modelo de linguagem para formar uma cadeia de execução
    chain = prompt | llm
    
    # Executa a cadeia passando a entrada do usuário e obtém a resposta
    response = chain.invoke({"input": state["input"]})
    
    # Retorna o conteúdo da resposta gerada pelo modelo de linguagem
    return {"output": response.content}
