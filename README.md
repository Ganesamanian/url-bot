# URL Bot

This is a simple URL bot built with **Streamlit** and **Langchain** that allows users to input URLs, fetches their contents, processes them, and provides answers to user queries based on the text extracted from those URLs.

## Features

- **URL Input:** Allows users to input up to 3 URLs.
- **Data Processing:** The bot loads the content from the provided URLs, chunks the data into manageable pieces, and generates embeddings for efficient retrieval.
- **Question-Answering:** Users can ask questions, and the bot uses a generative AI model to provide answers based on the text content of the URLs.
- **Embedding and Indexing:** The text content is processed and indexed using **FAISS** to provide fast and efficient query results.

## Requirements

Make sure you have the following libraries installed:

- `streamlit`
- `pickle`
- `langchain`
- `google-generativeai`
- `langchain_google_genai`
- `langchain_huggingface`
- `FAISS`
- `HuggingFaceEmbeddings`

   You can install the necessary libraries by running:
   ```bash
   pip install streamlit langchain google-generativeai langchain_google_genai langchain_huggingface faiss-cpu


## Setup

Follow these steps to get the project running locally:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/url-bot.git
   cd url-bot

2. **Install the required packages**:
   ```bash
   pip install -r requirements.txt

3. **Set up your Google API key**:

   - Go to the Google Cloud Console.
   - Create a new project and enable the Gemini API.
   - Set the environment variable for the API key:
   ```bash
   export GOOGLE_API_KEY="your_google_api_key"

4. **Run the Streamlit app**:
   ```bash
   streamlit run main.py

The app will open in your browser, and you can begin using it to find universities offering your chosen course.

# How it Works

### 1. Enter URLs:
In the sidebar, you can input up to three URLs. The bot will process the content of these URLs and convert them into chunks for processing.

### 2. Data Loading and Text Splitting:
The content of the URLs is loaded using the `UnstructuredURLLoader` from Langchain. The text is then split into smaller chunks for efficient processing and embedding.

### 3. Embeddings and Vector Index:
The text chunks are embedded into vector space using the `HuggingFaceEmbeddings` model (`sentence-transformers/all-mpnet-base-v2`). These embeddings are indexed using FAISS for fast retrieval during the query process.

### 4. Querying:
You can ask the bot questions related to the content of the URLs. The bot uses the generative AI model (`gemini-1.5-flash`) to provide answers based on the indexed content.

### 5. Answering:
The bot retrieves relevant information from the indexed text and generates an answer to the question.

# Code Overview

### URL Processing:
- The bot accepts up to 3 URLs via the Streamlit sidebar.
- The `UnstructuredURLLoader` loads the content of the URLs.
- The text is split into chunks using the `RecursiveCharacterTextSplitter`.

### Embedding & Indexing:
- The `HuggingFaceEmbeddings` model is used to generate embeddings for the text chunks.
- These embeddings are stored in a FAISS index.

### Query Answering:
- The user can input a query, and the bot retrieves relevant information from the indexed text using `RetrievalQAWithSourcesChain`.


## License

This project is licensed under the **MIT License**. See the [LICENSE](./LICENSE) file for details.

## Contribution

Feel free to contribute by opening issues or submitting pull requests. If you have any questions or suggestions, don't hesitate to reach out!





 
