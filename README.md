# QA Document Chatbot

A FastAPI-based question-answering chatbot for PDF documents, utilizing Retrieval-Augmented Generation (RAG) with agentic routing, multilingual support, and conversation memory. The chatbot ingests PDFs, generates embeddings, stores metadata in MongoDB, and answers queries based on document content.

## Description

This project is a document-based QA chatbot that enables users to upload PDF files, query their contents, and receive accurate responses. It supports multiple documents, remembers conversation history, and leverages advanced RAG techniques for efficient retrieval. The application is dockerized for seamless deployment and uses Google's Gemini-1.5-Flash for language modeling and EmbeddingGemma for multilingual embeddings.

## Features

- Upload and process multiple PDF documents.
- Generate document summaries and embeddings for efficient search.
- Agentic RAG pipeline with Maximum Marginal Relevance (MMR) ranking for improved retrieval.
- Conversation memory to retain up to 6 previous chat turns.
- Multilingual support for over 100 languages (via EmbeddingGemma).
- Store document metadata (file name, size, pages, description) in MongoDB.
- Delete all files and vectorstores (`/delete`) to clear storage and reset state.
- Dockerized for easy setup and deployment.

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


