# Sistema de Recuperação de Informações com FAISS e RetrievalQA

## Descrição

Este projeto implementa um sistema de perguntas e respostas baseado em Recuperação Aumentada por Geração (RAG), utilizando Python, LangChain, FAISS e Ollama.

O sistema utiliza uma base de conhecimento sobre Redes de Computadores, permitindo que o usuário faça perguntas em linguagem natural e receba respostas fundamentadas nos documentos recuperados.

## Tecnologias utilizadas

- Python
- LangChain
- FAISS
- Ollama
- mxbai-embed-large — modelo de embeddings
- llama3.2 — modelo de linguagem
- RetrievalQA — pipeline de recuperação e geração de respostas

## Funcionamento

1. O sistema lê o arquivo `documentos/redes.txt`.
2. O conteúdo é dividido em chunks utilizando `RecursiveCharacterTextSplitter`.
3. Cada chunk é convertido em um vetor numérico pelo modelo `mxbai-embed-large`.
4. Os embeddings são armazenados em um índice vetorial FAISS.
5. Quando o usuário faz uma pergunta, ela também é convertida em embedding.
6. O FAISS recupera os documentos semanticamente mais próximos.
7. O RetrievalQA envia os documentos recuperados ao modelo `llama3.2`, que gera a resposta.

## Requisitos

- Python 3.10 ou superior, em versão compatível com as dependências
- Ollama instalado
- Modelos `mxbai-embed-large` e `llama3.2` disponíveis localmente

## Instalação

Clone o repositório:

```bash
git clone URL_DO_REPOSITORIO
cd rag-faiss
```

Crie e ative o ambiente virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Instale as dependências:

```bash
python -m pip install -r requirements.txt
```

Instale o Ollama seguindo as instruções oficiais em https://ollama.com/download.

Baixe os modelos necessários:

```bash
ollama pull mxbai-embed-large
ollama pull llama3.2
```

Verifique se o serviço Ollama está em execução.

## Execução

Primeiro, crie o banco de dados vetorial:

```bash
python criar_banco.py
```

O programa processará o documento, gerará os embeddings e salvará o índice na pasta `faiss_index`.

Em seguida, execute o sistema interativo:

```bash
python main.py
```

Digite perguntas no terminal. Para encerrar, digite `sair`.

## Exemplos de perguntas e respostas

**Pergunta:** Qual protocolo garante a entrega confiável dos dados?

**Resposta:** TCP (Transmission Control Protocol).

**Pergunta:** Para que serve o DNS?

**Resposta esperada:** O DNS permite resolver nomes de domínio em endereços IP.

**Pergunta:** O que significa DORA no DHCP?

**Resposta esperada:** Discover, Offer, Request e Acknowledgment.

As respostas podem variar conforme o modelo de linguagem.

## Estrutura do projeto

```text
rag-faiss/
├── documentos/
│   └── redes.txt
├── criar_banco.py
├── main.py
├── requirements.txt
└── README.md
```

O diretório `faiss_index` é gerado durante a execução de `criar_banco.py`.

## Considerações finais

O projeto demonstra a integração entre embeddings, armazenamento vetorial e modelos de linguagem para construção de sistemas de recuperação de informações.

A utilização do FAISS permite localizar documentos por similaridade semântica, enquanto o RetrievalQA integra os documentos recuperados à geração de respostas.

A qualidade das respostas depende da base de conhecimento, da estratégia de recuperação e do modelo de linguagem utilizado.

## Referências

- LangChain: https://docs.langchain.com/
- FAISS: https://github.com/facebookresearch/faiss
- Ollama: https://ollama.com/
- Notebook de referência: https://github.com/tmedeirosb/tsi-llm/blob/main/vb-faiss-un04.ipynb

