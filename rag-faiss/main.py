
from pathlib import Path

from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA

# 1. Configurar o modelo de embeddings
embeddings = OllamaEmbeddings(
    model="mxbai-embed-large"
)

# 2. Carregar o banco vetorial existente
if not Path("faiss_index").exists():
    raise FileNotFoundError(
        "Banco FAISS não encontrado. Execute criar_banco.py primeiro."
    )

vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# 3. Configurar o modelo de linguagem
llm = OllamaLLM(
    model="llama3.2",
    temperature=0
)

# 4. Configurar a recuperação de documentos
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# 5. Criar o pipeline RetrievalQA
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# 6. Sistema interativo
print("\n=== SISTEMA DE PERGUNTAS E RESPOSTAS ===")
print("Digite 'sair' para encerrar.\n")

while True:
    pergunta = input("Pergunta: ").strip()

    if pergunta.lower() == "sair":
        print("Encerrando o sistema...")
        break

    if not pergunta:
        continue

    resultado = qa.invoke({"query": pergunta})

    print("\nResposta:")
    print(resultado["result"])

    print("\nDocumentos recuperados:")
    for indice, documento in enumerate(
        resultado["source_documents"], start=1
    ):
        print(f"\n--- Documento {indice} ---")
        print(documento.page_content)

    print("\n" + "-" * 50 + "\n")

