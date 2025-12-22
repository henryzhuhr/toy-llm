"""
Build a custom RAG agent with LangGraph
https://docs.langchain.com/oss/python/langgraph/agentic-rag

Split markdown:
https://docs.langchain.com/oss/python/integrations/splitters/markdown_header_metadata_splitter
"""

import os
from datetime import datetime
from typing import Literal, Optional

from langchain.tools import tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from loguru import logger
from pydantic import BaseModel, Field

print("Loading documents...")
urls = [
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
]

docs = [WebBaseLoader(url).load() for url in urls]
# print(repr(docs[0][0].page_content.strip()[:100]))

print("Splitting documents...")
docs_list = [item for sublist in docs for item in sublist]
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs_list)
# print(repr(doc_splits[0].page_content.strip()))


vectorstore = InMemoryVectorStore.from_documents(
    documents=doc_splits, embedding=OllamaEmbeddings(model="qwen3-embedding:0.6b")
)
retriever = vectorstore.as_retriever()


@tool
def retrieve_blog_posts(query: str) -> str:
    """Search and return information about Lilian Weng blog posts."""
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])


retriever_tool = retrieve_blog_posts

retriever_tool.invoke({"query": "types of reward hacking"})


base_url = "http://localhost:11434"
model_name: str = "qwen3:1.7b"
chat_model = ChatOllama(base_url=base_url, model=model_name)


def generate_query_or_respond(state: MessagesState):
    """Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
    """
    response = chat_model.bind_tools([retriever_tool]).invoke(state["messages"])
    return {"messages": [response]}


# input = {"messages": [{"role": "user", "content": "hello!"}]}
input = MessagesState(
    messages=[
        HumanMessage(
            content="What are the different types of reward hacking discussed in Lilian Weng's blog?"
        )
    ]
)
generate_query_or_respond(input)["messages"][-1].pretty_print()


"""
========================
Grade documents
========================
"""

GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n "
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
)
GRADE_PROMPT = (
    "您是一个评估检索到的文档与用户问题相关性的评分员。 \n "
    "以下是检索到的文档: \n\n {context} \n\n"
    "以下是用户问题: {question} \n"
    "如果文档包含与用户问题相关的关键词或语义意义，则将其评为相关，返回 `yes`，如果不相关返回 `no`。 \n"
)


class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""

    binary_score: str = Field(
        # description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
        description="描述文档是否与问题相关，'yes' 表示相关，'no' 表示不相关"
    )


base_url = "http://localhost:11434"
model_name: str = "qwen3:1.7b"
grader_model = ChatOllama(base_url=base_url, model=model_name)


def grade_documents(
    state: MessagesState,
) -> Literal["generate_answer", "rewrite_question"]:
    """Determine whether the retrieved documents are relevant to the question."""
    question = state["messages"][0].content
    context = state["messages"][-1].content

    prompt = GRADE_PROMPT.format(question=question, context=context)
    logger.info(f"🤖 Grader prompt: {prompt}")
    structured_output_model = grader_model.with_structured_output(
        GradeDocuments, include_raw=True
    )
    response: GradeDocuments = structured_output_model.invoke(
        [HumanMessage(content=prompt)],
    )  # type: ignore

    # 安全提取 parsed 对象
    if not isinstance(response, dict):
        logger.error("Unexpected response type from grader model")
        return "rewrite_question"  # 或抛异常

    parsed_response: Optional[GradeDocuments] = response.get("parsed")
    parsing_error = response.get("parsing_error")

    if parsing_error:
        logger.error(f"Parsing error: {parsing_error}")
        return "rewrite_question"

    if parsed_response is None:
        logger.warning("Parsed result is None, treating as not relevant")
        return "rewrite_question"

    print(f"Grader response: ({type(response)}) {response}")
    score = parsed_response.binary_score

    if score == "yes":
        return "generate_answer"
    else:
        return "rewrite_question"


"""
========================
Rewrite question
========================
"""

REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Formulate an improved question:"
)
REWRITE_PROMPT = (
    "观察输入并尝试推理其背后的语义意图 `/` 含义。\n"
    "这里是初始问题："
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "提出一个改进的问题:"
)


def rewrite_question(state: MessagesState):
    """Rewrite the original user question."""
    messages = state["messages"]
    question = messages[0].content
    prompt = REWRITE_PROMPT.format(question=question)
    response = chat_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [HumanMessage(content=response.content)]}


"""
========================
Generate an answer
========================
"""
GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question} \n"
    "Context: {context}"
)
GENERATE_PROMPT = (
    "您是问答任务的助手。 "
    "使用以下检索到的上下文来回答问题。 "
    "如果你不知道答案，就说我不知道。 "
    "使用最多三句话，并保持回答简洁。\n"
    "问题: {question} \n"
    "上下文: {context}"
)


def generate_answer(state: MessagesState):
    """Generate an answer."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = chat_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}


workflow = StateGraph(MessagesState)

# Define the nodes we will cycle between
workflow.add_node(generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node(rewrite_question)
workflow.add_node(generate_answer)

workflow.add_edge(START, "generate_query_or_respond")

# Decide whether to retrieve
workflow.add_conditional_edges(
    "generate_query_or_respond",
    # Assess LLM decision (call `retriever_tool` tool or respond to the user)
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retrieve",
        END: END,
    },
)

# Edges taken after the `action` node is called.
workflow.add_conditional_edges(
    "retrieve",
    # Assess agent decision
    grade_documents,
)
workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")

# Compile
graph = workflow.compile()


# graph_mermaid = graph.get_graph().draw_mermaid()
# os.makedirs("tmp", exist_ok=True)
# with open("tmp/test.md", "wb") as f:
#     f.write(f"{datetime.now()}\n```mermaid\n{graph_mermaid}\n```".encode())

for chunk in graph.stream(
    {"messages": [HumanMessage(content="Lilian Weng对奖励黑客的类型有什么看法？")]},
    debug=True,
):
    for node, update in chunk.items():
        print("Update from node", node)
        update["messages"][-1].pretty_print()
        print("\n\n")
