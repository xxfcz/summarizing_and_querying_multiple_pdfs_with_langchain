# pip install langchain openai chromadb tiktoken

from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import PyPDFDirectoryLoader

loader = PyPDFDirectoryLoader("./pdfs/")

docs = loader.load()

# Create the vector store index
index = VectorstoreIndexCreator().from_loaders([loader])

query = "What is the core idea behind the CoOP (context optimization) paper?"

ans = index.query(query)
print(ans)
#  The core idea behind the CoOP paper is to model a prompt's context words with learnable vectors while keeping the entire pre-trained parameters fixed, in order to adapt CLIP-like vision-language models for downstream image recognition tasks.
