import dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents import (
    create_openai_tools_agent,
    Tool,
    AgentExecutor,
)
from langchain import hub
from worker_tools import get_current_wait_time

CHROMA_PATH = "chroma_data/"

chat_model = None

dotenv.load_dotenv()

chat_template_str = """Your job is to answer questions that asking how to do something.
Do not answer What, Who, When, Why questions, said you don't know and ask user to ask a question related.
Use the following context to answer questions.
Be as detailed as possible, but don't make up any information that's not from the context.
If you don't know an answer, say you don't know and do not provide any information or suggestion.

{context}
"""

chat_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["context"],
        template=chat_template_str,
    )
)

chat_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["question"],
        template="{question}",
    )
)
messages = [chat_system_prompt, chat_human_prompt]

chat_prompt_template = ChatPromptTemplate(
    input_variables=["context", "question"],
    messages=messages,
)

chat_model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

chat_vector_db = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=OpenAIEmbeddings()
)

chat_retriever  = chat_vector_db.as_retriever(k=10)

chat_chain = (
    {"context": chat_retriever, "question": RunnablePassthrough()}
    | chat_prompt_template
    | chat_model
    | StrOutputParser()
)

tools = [
    Tool(
        name="Question",
        func=chat_chain.invoke,
        description="""Always use this tool.
        with only exception for questions about wait time for worker.
        Pass the entire question as input to the tool. For instance,
        if the question is "How to Get Popular on Instagram?",
        the input should be "How to Get Popular on Instagram?"
        """,
    ),
    Tool(
        name="Waits",
        func=get_current_wait_time,
        description="""Use when asked about wait time for worker to help.
        This tool can only get the wait time for worker now.
        This tool returns wait times in minutes.
        Do not pass the word "worker" as input, only the worker name itself.
        For instance, if the question is
        "What is the current wait time for worker marco?", the input should be "marco".
        """,
    ),
]

agent_prompt = hub.pull("hwchase17/openai-tools-agent")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You must pass the input to agent tools. Pass all the intermediate result directly. If you don't have information, say you don't have information""",),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name='chat_history', optional=True),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent_prompt.messages = prompt.messages

agent_chat_model = ChatOpenAI(
    model="gpt-3.5-turbo-1106",
    temperature=0
)

chat_agent = create_openai_tools_agent(
    llm=agent_chat_model,
    prompt=agent_prompt,
    tools=tools,
)

chat_agent_executor = AgentExecutor(
    agent=chat_agent,
    tools=tools,
    return_intermediate_steps=True,
    verbose=True,
)