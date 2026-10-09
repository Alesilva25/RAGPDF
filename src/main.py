from langchain_chroma.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

DB_PATH = 'db'

prompt_template = """
Responda a pergunta do usuário:
{question}

com base nessas informações abaixo:
{knowledge_base}

Se você não encontrar a resposta para a pergunta nessas informações, responda 'não sei'
"""

def ask():
    question = input("Escreva a sua pergunta: ")

    # load database
    function = OpenAIEmbeddings()
    knowledge_base = Chroma(
        persist_directory=DB_PATH,
        embedding_function=function
        )

    # compare user's question to database
    results = knowledge_base.similarity_search_with_relevance_scores(question, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        print("Não consegui encontrar informação relevante na base") 
        return

    result_text = [result[0].page_content for result in results]
    knowledge_base = "\n\n----\n\n".join(result_text)

    prompt = ChatPromptTemplate.from_template(prompt_template)
    prompt = prompt.invoke(
        {
            "question" : question,
            "knowledge_base" : knowledge_base
        }
    )

    model = ChatOpenAI()

    answer = model.invoke(prompt).content
    print(f"Resposta da IA: {answer}")

ask()