"""
Utility Functions for RAG (Retrieval-Augmented Generation) System
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai.chat_models import ChatOpenAI
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from typing import List, Dict, Any

def preprocess_text(text: str) -> List[str]:
    """
    Preprocesses text by converting to lowercase, removing special characters and numbers,
    tokenizing, removing stop words, and lemmatizing.
    
    Args:
    text (str): The input text to preprocess
    
    Returns:
    List[str]: A list of preprocessed tokens
    """
    if isinstance(text, str):
        # Convert text to lowercase
        text = text.lower()
        
        # Remove special characters, numbers, and punctuation
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Remove stop words
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        
        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        
        return tokens
    else:
        return []

def text_to_vectors(text: List[str], word2vec_model: Word2Vec) -> List[Any]:
    """
    Converts a list of preprocessed tokens to vector representations using Word2Vec.
    
    Args:
    text (List[str]): A list of preprocessed tokens
    word2vec_model (Word2Vec): The trained Word2Vec model
    
    Returns:
    List[Any]: A list of vector representations for the input tokens
    """
    vectors = []
    for word in text:
        if word in word2vec_model.wv:
            vectors.append(word2vec_model.wv[word])
    return vectors

def create_passage_dict_with_vectors(data: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    """
    Creates a dictionary of passages and their vector representations.
    
    Args:
    data (pd.DataFrame): DataFrame containing the message and explanation data
    
    Returns:
    Dict[int, Dict[str, Any]]: Dictionary mapping message IDs to dictionaries containing
                              message text, explanation text, message vectors, and explanation vectors
    """
    passage_dict = {}
    for index, row in data.iterrows():
        message_id = index
        message = row['Phishing/scam message']
        explanation = row['Explanations and CTA']
        message_vectors = row['Message Vectors']
        explanation_vectors = row['Explanation Vectors']

        passage_dict[message_id] = {
            'message': message,
            'explanation': explanation,
            'message_vectors': message_vectors,
            'explanation_vectors': explanation_vectors
        }
    return passage_dict

def query_knowledge_base(query: str, knowledge_base: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Simulates querying the knowledge base to retrieve relevant passages.
    
    Args:
    query (str): The input query
    knowledge_base (List[Dict[str, Any]]): List of dictionaries representing the knowledge base
    
    Returns:
    List[Dict[str, Any]]: A list of dictionaries containing the retrieved passages
    """
    # For this example, we'll simply return all passages. In a real implementation,
    # you would use some form of semantic similarity or relevance scoring to determine
    # which passages are most relevant to the query.
    return knowledge_base

def generate_responses(query: str, conversation_history: List[HumanMessage], 
                      knowledge_base: List[Dict[str, Any]], chat_model: ChatOpenAI) -> List[HumanMessage]:
    """
    Generates responses using the RAG system.
    
    Args:
    query (str): The input query
    conversation_history (List[HumanMessage]): List of messages in the current conversation
    knowledge_base (List[Dict[str, Any]]): List of dictionaries representing the knowledge base
    chat_model (ChatOpenAI): The OpenAI chat model instance
    
    Returns:
    List[HumanMessage]: A list of HumanMessage objects containing the generated responses
    """
    relevant_passages = query_knowledge_base(query, knowledge_base)
    
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=query)
    ]
    messages.extend(conversation_history)

    responses = chat_model.invoke(messages)
    return responses

def generate_responses_with_context(query: str, conversation_history: List[HumanMessage],
                                   knowledge_base: List[Dict[str, Any]], chat_model: ChatOpenAI) -> List[HumanMessage]:
    """
    Generates responses with context understanding using the RAG system.
    
    Args:
    query (str): The input query
    conversation_history (List[HumanMessage]): List of messages in the current conversation
    knowledge_base (List[Dict[str, Any]]): List of dictionaries representing the knowledge base
    chat_model (ChatOpenAI): The OpenAI chat model instance
    
    Returns:
    List[HumanMessage]: A list of HumanMessage objects containing the generated responses
    """
    relevant_passages = query_knowledge_base(query, knowledge_base)
    
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=query)
    ]
    messages.extend(conversation_history)

    responses = chat_model.invoke(messages)
    return responses
