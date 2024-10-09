# Importa a biblioteca pdfplumber, que permite a extração de texto de arquivos PDF
import pdfplumber

# Importa o CharacterTextSplitter da LangChain, que permite dividir o texto em partes menores (chunks)
from langchain.text_splitter import CharacterTextSplitter

# Importa HuggingFaceEmbeddings da LangChain, que utiliza modelos de embeddings pré-treinados do Hugging Face
from langchain_huggingface import HuggingFaceEmbeddings

# Importa Chroma, um banco de dados de vetores da LangChain, para armazenar os embeddings dos documentos
from langchain_chroma import Chroma

# Importa a classe Document da LangChain, que encapsula o conteúdo de um documento
from langchain.docstore.document import Document

# Função para extrair texto de um PDF
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extrai texto de um arquivo PDF e retorna como uma string completa."""
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            full_text += page.extract_text()
    return full_text

# Função para converter o texto extraído em um objeto Document
def convert_to_document(text: str) -> Document:
    """Converte o texto fornecido em um objeto Document da LangChain."""
    return Document(page_content=text)

# Função para dividir o texto em chunks
def split_text_into_chunks(document: Document, chunk_size: int = 1250, overlap: int = 100):
    """Divide o texto do documento em partes menores (chunks) com o tamanho e sobreposição definidos."""
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, separator="\n", chunk_overlap=overlap)
    return text_splitter.split_documents([document])

# Função para criar embeddings e vetorializar os documentos
def create_vector_db(text_chunks, model_name: str, persist_directory: str):
    """Cria embeddings e armazena os documentos vetorizados no banco de dados Chroma."""
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vector_db = Chroma.from_documents(text_chunks, embeddings, persist_directory=persist_directory)
    print("Documents vectorized.")
    return vector_db

# Função principal para orquestrar o fluxo de extração, vetorização e armazenamento
def main():
    # Caminho do PDF
    pdf_path = "documentos/L14133.pdf"
    # Diretório onde o banco de dados de vetores será armazenado
    persist_directory = "Vector_DB_directory"
    # Modelo de embeddings Hugging Face
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Passo 1: Extrair o texto do PDF
    full_text = extract_text_from_pdf(pdf_path)
    
    # Passo 2: Converter o texto extraído em um objeto Document
    document = convert_to_document(full_text)
    
    # Passo 3: Dividir o texto em chunks
    text_chunks = split_text_into_chunks(document)
    
    # Passo 4: Criar embeddings e vetorializar os documentos
    vector_db = create_vector_db(text_chunks, model_name, persist_directory)
    
    
# Ponto de entrada do script
if __name__ == "__main__":
    main()





