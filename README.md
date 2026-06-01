# Production AI Assistant with Memory + RAG

## Overview
This project is a production-level AI assistant that combines Retrieval-Augmented Generation (RAG) with conversational memory. The assistant can answer questions from custom documents while maintaining conversation context across multiple interactions.

## Features
- Conversational AI using LLMs
- Retrieval-Augmented Generation (RAG)
- Persistent conversation memory
- PDF document processing
- Semantic search using vector embeddings
- Context-aware responses

## Tech Stack
- Python
- LangChain
- Ollama (Llama 3.2)
- FAISS Vector Database
- PyPDFLoader
- RecursiveCharacterTextSplitter

## Project Workflow
1. Load PDF documents
2. Split documents into chunks
3. Generate embeddings
4. Store embeddings in FAISS
5. Retrieve relevant chunks
6. Maintain conversation memory
7. Generate context-aware answers

## Skills Demonstrated
- Generative AI
- RAG Architecture
- Vector Databases
- Embedding Models
- Conversational Memory
- LangChain Development

## Installation

```bash
pip install langchain langchain-community langchain-classic faiss-cpu pypdf
