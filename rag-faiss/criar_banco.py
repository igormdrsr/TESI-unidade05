from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_classic.chains import RetrievalQA

from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 1. Ler o conteúdo do arquivo
caminho = Path("documentos/redes.txt")
texto = caminho.read_text(encoding="utf-8")

# 2. Transformar o texto em um documento do LangChain
documento = Document(
    page_content=texto,
    metadata={"source": str(caminho)}
)

# 3. Configurar a divisão em chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    length_function=len
)

# 4. Dividir o documento
chunks = text_splitter.split_documents([documento])

# 5. Exibir os chunks gerados
print(f"Total de chunks: {len(chunks)}")

for indice, chunk in enumerate(chunks, start=1):
    print(f"\n--- Chunk {indice} ---")
    print(chunk.page_content)

# 6. Configurar o modelo de embeddings
embeddings = OllamaEmbeddings(
    model="mxbai-embed-large"
)

# 7. Gerar os embeddings e criar o banco vetorial
vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

# 8. Salvar o banco vetorial no disco
vectorstore.save_local("faiss_index")

print("\nBanco vetorial FAISS criado e salvo com sucesso!")
print(f"Total de vetores armazenados: {vectorstore.index.ntotal}")

# Realizar busca semântica com pontuação
pergunta = "Qual protocolo garante a entrega confiável dos dados?"

resultados = vectorstore.similarity_search_with_score(
    query=pergunta,
    k=5
)

print(f"\nPergunta: {pergunta}")

for indice, (documento, distancia) in enumerate(resultados, start=1):
    print(f"\n--- Resultado {indice} ---")
    print(f"Distância: {distancia:.4f}")
    print(documento.page_content)


# 11. Configurar o modelo de linguagem
llm = OllamaLLM(
    model="llama3.2",
    temperature=0
)

# 12. Configurar o recuperador de documentos
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# 13. Criar o pipeline RetrievalQA
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# 14. Fazer uma pergunta ao sistema
pergunta = "Qual protocolo garante a entrega confiável dos dados?"

resultado = qa.invoke({"query": pergunta})

print("\n=== RESPOSTA DO SISTEMA ===")
print(resultado["result"])

print("\n=== DOCUMENTOS UTILIZADOS ===")

for indice, documento in enumerate(
    resultado["source_documents"], start=1
):
    print(f"\nDocumento {indice}:")
    print(documento.page_content)

