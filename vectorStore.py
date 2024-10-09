# Importa HuggingFaceEmbeddings da LangChain, que utiliza modelos de embeddings pré-treinados da Hugging Face
from langchain_huggingface import HuggingFaceEmbeddings
# Importa Chroma, que é um banco de dados de vetores da LangChain, usado para armazenar e manipular embeddings vetorizados
from langchain_chroma import Chroma

# Carregar o banco de dados Chroma
persist_directory = "Vector_DB_directory"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# Certifique-se de que a variável esteja definida corretamente
crhomadb = Chroma(
    persist_directory="Vector_DB_directory",
    embedding_function=embeddings
)