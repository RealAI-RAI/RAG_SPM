"""
Main Script for RAG_SPM (Retrieval-Augmented Generation for Phishing and Scam Message Analysis)
"""

import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai.chat_models import ChatOpenAI
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from RAG_SPM.preprocessing import preprocess_text, text_to_vectors
from RAG_SPM.rag_utils import create_passage_dict_with_vectors, query_knowledge_base, generate_responses, generate_responses_with_context

# Load environment variables
load_dotenv()

# Get OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

# Initialize ChatOpenAI model
model = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model="gpt-3.5-turbo"
)

def main():
    # Load data
    file_path = "/content/PHISHING & SCAM MESSAGES ONLY + THEIR SPECIFIC EXPLANATIONS IN OUR DESIRED FORMAT (1).xlsx"
    data = pd.read_excel(file_path)

    # Preprocess text data
    data['Phishing/scam message'] = data['Phishing/scam message'].apply(preprocess_text)
    data['Explanations and CTA'] = data['Explanations and CTA'].apply(preprocess_text)

    # Train Word2Vec model
    def train_word2vec_model(data):
        sentences = data['Phishing/scam message'].tolist() + data['Explanations and CTA'].tolist()
        word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=1)
        return word2vec_model

    word2vec_model = train_word2vec_model(data)

    # Convert preprocessed text to vector representations
    data['Message Vectors'] = data['Phishing/scam message'].apply(lambda text: text_to_vectors(text, word2vec_model))
    data['Explanation Vectors'] = data['Explanations and CTA'].apply(lambda text: text_to_vectors(text, word2vec_model))

    # Create knowledge base
    def create_passage_dict_with_vectors(data):
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

    passage_dict = create_passage_dict_with_vectors(data)

    # Generate responses
    def generate_responses_with_context(query, conversation_history, knowledge_base, chat_model):
        relevant_passages = query_knowledge_base(query, knowledge_base)
        
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=query)
        ]
        messages.extend(conversation_history)

        responses = chat_model.invoke(messages)
        return responses

    # Example usage
    query = "How can I protect my university account from phishing?"
    conversation_history = [
        HumanMessage(content="I received an email asking for my login credentials. Is it safe to provide them?"),
        SystemMessage(content="No, university IT departments typically do not ask for login credentials via email.")
    ]
    
    responses = generate_responses_with_context(query, conversation_history, passage_dict, model)

    # Print generated responses
    for i, response in enumerate(responses):
        print(f"Response {i+1}: {response}")

if __name__ == "__main__":
    main()
