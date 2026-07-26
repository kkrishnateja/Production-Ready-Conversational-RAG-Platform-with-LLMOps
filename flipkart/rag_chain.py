from langchain_groq import ChatGroq
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from flipkart.config import Config


class RAGChainBuilder:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.model = ChatGroq(model= Config.RAG_MODEL, temperature=0.5)
        self.history_store = {}

    def _get_history(self, session_id:str) -> BaseChatMessageHistory: # Private method
        if session_id not in self.history_store:
            self.history_store[session_id] = ChatMessageHistory()

        return self.history_store[session_id]
    
    def build_chain(self):
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3}) # No of relevant docs to be returned

        context_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given the chat history and user question, rewrite it as a standalone question."),
            MessagesPlaceholder(variable_name="chat_history"), 
            ("human", "{input}")  
        ]) # Rewrites the user question using the previous context(Internally)

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """You're an e-commerce bot answering product-related queries using reviews and titles.
                          Stick to context. Be concise and helpful.\n\nCONTEXT:\n{context}\n\nQUESTION: {input}"""), # This is the changed input
            MessagesPlaceholder(variable_name="chat_history"), 
            ("human", "{input}")  # User input
        ])

        history_aware_retriever = create_history_aware_retriever(
            self.model, retriever, context_prompt
        )

        question_answer_chain = create_stuff_documents_chain(
            self.model, qa_prompt
        )

        rag_chain = create_retrieval_chain(
            history_aware_retriever, question_answer_chain
        )

        return RunnableWithMessageHistory(
            rag_chain,
            self._get_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer" # By default it is answer only
        )