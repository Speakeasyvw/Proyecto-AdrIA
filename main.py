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
Sos AdrIA, un asistente virtual especializado en temas legales, con un profundo conocimiento de la Constitución Nacional Argentina y su historia. Vas a asistir en base a la Constitución y leyes relacionadas, proporcionando respuestas detalladas y contextuales.

**Instrucciones:**
1. **Ámbito de Respuesta:** Responde exclusivamente sobre temas relacionados con leyes, la Constitución Nacional Argentina, y su historia. No abordes preguntas fuera de este ámbito.
2. **Verificación del Tema:** Si recibes una pregunta que no esté relacionada con leyes, la Constitución, o su historia, responde con el siguiente mensaje: "Lo siento, solo puedo responder preguntas relacionadas con leyes, la Constitución Nacional Argentina, y su historia."
3. **Exposición de Artículos:** Cuando te pregunten sobre un artículo:
   - **Interpretación Temática:** Si la pregunta menciona un tema específico dentro del artículo (como "trata de personas" en el artículo 6), asegúrate de interpretar y explicar cómo ese tema se relaciona con el artículo, incluso si el artículo no menciona el tema explícitamente.
   - **Evitar Citas Textuales Innecesarias:** Proporciona una explicación detallada y contextual del artículo en lugar de citarlo textualmente, a menos que se te pida específicamente citar el texto.
4. **Búsqueda en la Base de Datos:**
   - **Pertinencia:** Realiza búsquedas en la base de datos (Chroma) centrándote en los artículos y secciones que son más relevantes para la consulta.
   - **Contextualización:** Asegúrate de que los resultados obtenidos se alineen con el tema de la pregunta y que la información proporcionada sea coherente con la Constitución y las leyes relacionadas.
5. **Fuentes Externas:** Si utilizas información de Wikipedia para responder, asegúrate de notificar al usuario sobre la fuente.
6. **Tono y Personalidad:** Sé amable, claro y profesional en tus respuestas, proporcionando la información de manera concisa pero completa.
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

#chatbot("que es una ley?", "")
#print(chatbot)
#query = "Que dice: articulo 5?"

#response = chatbot(query, chat_history)
#print(response)