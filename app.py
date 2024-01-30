from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.documents.base import Document
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langfuse.callback import CallbackHandler
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.schema import HumanMessage
import json
import re
import streamlit as st

class Tool:
    def __init__(self, name, func, description):
        self.name = name
        self.func = func
        self.description = description

class DiseaseQueryTool:
    def __init__(self, db):
        self.db = db
        self.query_type = "Sintomas"  # This should match the header in the markdown file

    def run(self, symptoms):
        # Process the documents to generate a response
        response = self.process_docs(symptoms)
        return {
            'user_prompt':symptoms,
            'system_prompt': f'Eres un modelo especialmente entrenado para responder preguntas sobre enfermedades y sintomas. El usuario te va a pasar un serie de sintomas, y quiero que en base a dichos sintomas me digas que enfermedades pueden ser de los siguientes datos:\n{response}.\nQuiero que unicamente te bases en esos datos para comprobar que enfermades podrian ser. Ahora te voy a pasar los sintomas del usuario y quiero que me digas todas las posibilidades que creas, no necesito que me digas las razones de por que pueden ser esas enfermedades, solo dime los nombres de las mismas. Y respondeme solamente en ESPAÑOL.\nSintomas del usuario:'
        } 
  
    def process_docs(self,query):
        docs = db.similarity_search(query) 
        for doc in docs:
            contemd=(doc.page_content)
        response = doc.page_content
        return response




class TreatmentQueryTool:
    def __init__(self, db):
        self.db = db

    def run(self, disease):
        # Use the RAG to find treatments for this disease
        query = "Tratamiento " + disease
        # Process the documents to generate a response
        response = self.process_docs(query)
        return {
            'user_prompt':disease,
            'system_prompt': f'Eres un modelo especialmente entrenado para responder preguntas sobre enfermedades y tratamientos. El usuario te va a pasar una enfermedad, y quiero que me digas el tratamiento necesario para curar esa enfermedad, QUIERO QUE ME DIGAS SOLAMENTE LO QUE VIENE EN LA SIGUIENTE INFORMACION:\n{response}.\nY respondeme solamente en ESPAÑOL.\nLa Enfermedad del usuario es la siguiente: '
        } 

    def process_docs(self,query):
        docs = db.similarity_search(query) 
        for doc in docs:
            contemd=(doc.page_content)
        response = doc.page_content
        return response

class FeelBetterQueryTool:
    def __init__(self, rag):
        self.rag = rag

    def run(self, disease):
        # Use the RAG to find ways to feel better when you have this disease
        query = disease + ", que hacer para sentirse mejor"
        # Process the documents to generate a response
        response = self.process_docs(query)
        return {
            'user_prompt':disease,
            'system_prompt': f'Eres un modelo especialmente entrenado para responder preguntas sobre enfermedades y que hacer para sentirse mejor. El usuario te va a pasar una enfermedad, y quiero que me digas que hacer para sentirse mejor mientras se tiene esa enfermedad. La informacion la tienes que sacar UNICA Y EXCLUSIVAMENTE de la siguiente informacion:\n{response}.\nY respondeme solamente en ESPAÑOL.\nLa Enfermedad del usuario es la siguiente:'
        } 

    def process_docs(self,query):
        docs = db.similarity_search(query) 
        for doc in docs:
            contemd=(doc.page_content)
        response = doc.page_content
        return response

class AgentTool:

    def __init__(self,query,llm):
        self.query = query
        self.llm = llm



class Agent:
    def __init__(self, llm, query, tools):
        self.tools = {tool.name: tool for tool in tools}
        self.llm = llm
        self.query = query

        
    def classificate_tool(self):
      
        classificate_agent_promt="""

                Eres un agente virtual especializado en el trat

                tu objetivo es clasificar el mensaje del usuario para poder proporcionar una funcion que se adapte a las necesidades del usuario, para ello tendras que repsponder con el identificador de la accion deseead

                Te paso un json que explica los tipos de funciones y sus identificadores 
                {

                    DiseaseQuery:"Esta es el identificador si el usuario te da una serie de sintomas y desear saber que enfermedad puede tener, si crees que esta es la opcion correcta responde con 'DiseaseQuery'".
                    TreatmentQuery:"Este es el identificador de la funcion, si el usario te da una enfermedad y desea saber el tratamiento que debe de seguir para mejorarse debes de responder con 'TreatmentQuery'",
                    FeelBetterQuery:"Esta es el identificador de la funcion que debes de responer si el usario quiere unas pautas para poder sentirse mejor mientras hace su tratamiento, si crees que esta es la funcion que deseea el usuario responde con 'FeelBetterQuery'"
                }

                A partir del prompt del usuario identifica el tipo de funcion deseaada y responde con su identificador dentro de un obejeto json de este modo:

                Ejemplo de uso 1:

                Prompt del usuario: Tengo sinusitis y quiero sentirme mejor
                Respuesta:
                {
                    "agent_tool": "FeelBetterQuery"
                }

                Ejemplo de uso 2:

                Prompt del usuario: Tengo tos y mocos
                Respuesta:
                {
                    "agent_tool": "DiseaseQuery"
                }


                Ejemplo de uso 3:

                Prompt del usuario: Que debo de hacer para curarme la gripe
                Respuesta:
                {
                    "agent_tool": "TreatmentQuery"
                }
                

                Recuerda solo responder solo con el objeto deseeado y ningun texto mas.

                """,




        agent_tool = self.llm.invoke([SystemMessage(content=classificate_agent_promt),HumanMessage(content=self.query)])

        tool_action = re.sub(r'\\|\\n', '', agent_tool.content)

        print(tool_action)

        tool_action = json.loads(tool_action)

        response = self.answer_query(tool_action['agent_tool'])

        return response

    def answer_query(self, query_type):
        if query_type in self.tools:
            try:
                # Use the appropriate tool to get the raw answer
                raw_answer = self.tools[query_type].func(self.query)
                # Check if raw_answer is not empty
                if raw_answer:
                    # Create a UserMessage with the user's query
                    human_message = HumanMessage(content=self.query)
                    # Use the language model to generate a human-readable response
                    response = self.llm.invoke([SystemMessage(content=raw_answer['system_prompt']),HumanMessage(content=raw_answer['user_prompt'])])
                else:
                    response = "Lo siento, no pude encontrar ninguna información relacionada con tu petición."
                return response
            except Exception as e:
                return f"An error occurred: {str(e)}"
        else:
            raise ValueError(f"No hay tool disponible para el tipo de tool {query_type}")


# Initialize the RAG
# Define the model name and parameters
model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}

# Create the HuggingFaceBgeEmbeddings object
hf = HuggingFaceBgeEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

markdown_files = ["db_file.md"]

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

documents=[]

for file in markdown_files:
    with open(file, 'r') as f:
        markdown_string = f.read()
    doc = markdown_splitter.split_text(markdown_string)
    documents+=doc

db = Chroma.from_documents(documents, hf, collection_metadata={
                                     "hnsw:space": "cosine"}, persist_directory="./db")

load_vector_store = Chroma(
    persist_directory="./db",embedding_function=hf)
rag = load_vector_store.as_retriever(search_kwargs={"k": 10})

# Initialize the tools
diseaseQueryTool = DiseaseQueryTool(db)
treatmentQueryTool = TreatmentQueryTool(db)
feelBetterQueryTool = FeelBetterQueryTool(db)

tools = [
    Tool(name="DiseaseQuery", func=diseaseQueryTool.run, description="Tool for symtoms to desea"),
    Tool(name="TreatmentQuery", func=treatmentQueryTool.run, description="Tool for desea to treatments"),
    Tool(name="FeelBetterQuery", func=feelBetterQueryTool.run, description="Tool for desea to how to feel better"),
]

# Initialize Langfuse handler
handler = CallbackHandler("pk-lf-2a4f28d9-5631-4bdf-9757-0243f5780f05", "sk-lf-dac01d3a-d461-4b36-8851-022af119b00c")

# Initialize the language model
llm = ChatOpenAI(
    base_url='https://api.together.xyz/',
    streaming = False,
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    temperature=0,
    max_tokens=1024,
    openai_api_key="acfe7357ec8ad88e6312ef6c5ec1f37f9a8b7f6c48cabf75bbd4b3e70d03b300",
    callbacks=[handler]
)

st.title("Asistente médico")

    # Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

    # Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

    # React to user input
if prompt := st.chat_input("¿Cuál es el problema?"):
        # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
        # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    agent = Agent(query=prompt,llm=llm,tools=tools)

    response = agent.classificate_tool()

        # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response.content)
        # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response.content})