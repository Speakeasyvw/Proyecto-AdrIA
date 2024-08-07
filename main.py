from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory, ChatMessageHistory
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate
from langchain import hub
from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from dotenv import load_dotenv

load_dotenv()

embeddings_model = OpenAIEmbeddings()

#cargo el vector store desde la persistencia
db = Chroma(
    persist_directory='docs/Chroma',
    embedding_function=embeddings_model
)

retriever = db.as_retriever(search_kwargs={"k": 20}) 

memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=4,  # Limitar el tamaño de la ventana de memoria a los últimos 4 mensajes
    return_messages=True,
    chat_memory=ChatMessageHistory()
)

#crear una tool de búsqueda
busqueda_constitucion = create_retriever_tool(
    retriever=retriever,
    name='busqueda_constitucion',
    description="Busca articulos o extractos en el archivo de la consticion nacional argentina 'constitucionar'"
)

tools = [busqueda_constitucion]


# Crear la herramienta de Wikipedia usando LangChain
wikipedia_api_wrapper = WikipediaAPIWrapper(lang="es")
busqueda_wikipedia = WikipediaQueryRun(
    api_wrapper=wikipedia_api_wrapper,
    name="Wikipedia",
    description="Busca información en Wikipedia en español."
)

tools = [busqueda_constitucion, busqueda_wikipedia]

# Cargar el agente desde el hub
prompt = hub.pull("hwchase17/openai-tools-agent")
prompt.messages[0].prompt.template = """
Sos un asistente Virtual amable llamado AdrIA, conoces la Constitucion Nacional como la palma de tu mano! Vas a asistir en base a la Constitución Nacional Argentina. 
Para cada consulta, busca los artículos relevantes y proporciona respuestas precisas basadas en ellos. Vas buscar por 'Articulos'.
En caso de no encontrar informacion en la constitucion nacional vas a acudir a la herremienta de busqueda de wikipedia como ultimo recurso o a menos que te lo especifiquen. 

Ejemplo de consulta: "¿Cómo se relacionan el artículo 22 y 23?".
Ejemplo de consulta 2: "¿Que es una ley?"


Vas a adoptar una personalidad amable y concisa, devolve respuestas en español a menos que te lo especifiquen.
Si en algun momento tenes error, disculpate y corregi tu respuesta.
En caso de sacar informacion de wikipedia, advertí sobre ello.
"""

# Crear el agente
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0
)

agent = create_openai_tools_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)

def chatbot(query:str, chat_history):
    response = agent_executor.invoke({'input': f'{query}', "chat_history":chat_history })['output']
    return response

chatbot("que es una ley?", "")
#print(chatbot)
#query = "Que dice: articulo 5?"

#response = chatbot(query, chat_history)
#print(response)