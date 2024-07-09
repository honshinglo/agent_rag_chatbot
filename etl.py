import dotenv
import os
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

CSV_PATH = "data"
CHROMA_PATH = "chroma_data"

dotenv.load_dotenv()

vectordb = Chroma(persist_directory=CHROMA_PATH, embedding_function=OpenAIEmbeddings())

for filename in os.listdir(CSV_PATH):
    file_path = os.path.join(CSV_PATH, filename)
    if filename.endswith(".csv"):
        loader = CSVLoader(file_path=file_path, source_column="title", encoding="utf8")
        reviews = loader.load()
        batch_size = 5461
        for i in range(0, len(reviews), batch_size):
            batch = reviews[i:i+batch_size]
            vectordb.add_documents(batch)