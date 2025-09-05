# Advanced Multi-Document RAG System with Agentic Routing

A production-ready Retrieval-Augmented Generation (RAG) system that intelligently routes queries across multiple document collections using advanced AI techniques. Built with LangChain, FAISS, and Google's Gemini LLM.

## Description

This project is a document-based QA chatbot that enables users to upload PDF files, query their contents, and receive accurate responses. It supports multiple documents, remembers conversation history, and leverages advanced RAG techniques for efficient retrieval. The application is dockerized for seamless deployment and uses Google's Gemini-1.5-Flash for language modeling and EmbeddingGemma for multilingual embeddings.

# Key Features
## 🎯 Intelligent Agentic Routing

- Multi-Agent Architecture: Separate Router Agent and RAG Agent for optimal performance
- Smart Document Selection: AI-powered routing to the most relevant document collection
- Graceful Fallbacks: Returns "I don't know" when no relevant documents are found

## 🧠 Advanced Retrieval Techniques

- Maximal Marginal Relevance (MMR): Ensures diverse, non-redundant context
- Optimized Parameters: k=10, fetch_k=20, lambda_mult=0.5 for balanced relevance-diversity
- Multi-Vectorstore Support: Independent FAISS indexes for different document types

## 💬 Conversational Memory

- Context-Aware: Maintains conversation history for follow-up questions
- Memory Buffer Management: Keeps last 6 conversation turns for optimal performance
- Structured Chat History: Clean JSON format for frontend integration

## 🛡️ Production-Ready Design

- Robust Error Handling: Never crashes, always returns structured responses
- Source Attribution: Precise citations with document names and page numbers
- Deterministic Responses: Temperature=0 for consistent, factual answers
- Comprehensive Logging: Built-in logging for debugging and monitoring


## Versions

### Version 1
- Used similarity search for document retrieval.
- Supported only single file input at a time.
- Lacked conversation memory; queries were independent.
- Provided a basic frontend for uploading and querying.

### Version 2
- Added MongoDB for storing document metadata (file name, size, pages, description).
- Dockerized the project for containerized deployment.
- Switched to MMR ranking for enhanced retrieval accuracy.
- Enabled multiple document uploads with agentic RAG for routing queries to the relevant vectorstore.
- Implemented conversation memory using LangChain to recall previous chats.
- Enhanced multilingual support with EmbeddingGemma for embeddings and Gemini-1.5-Flash for LLM tasks.
- Updated `/delete` endpoint to clear all files and vectorstores without requiring a file name.

## Installation

### Prerequisites
- Python 3.10+
- Docker (for containerized deployment)
- MongoDB (local or cloud instance, e.g., MongoDB Atlas)
- Google API Key for Gemini (set as `GEMINI_API_KEY` in `.env`)


