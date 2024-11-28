import pymupdf4llm
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

embed_model = OpenAIEmbedding(api_base="http://localhost:1234/v1", api_key="key", model_name="text-embedding-nomic-embed-text-v1.5@q8_0")
llm = OpenAI(api_base="http://localhost:1234/v1", api_key="key", model_name="mistral-nemo-instruct-2407")

Settings.llm = llm
Settings.embed_model = embed_model


#Load the Documents
llama_reader = pymupdf4llm.LlamaMarkdownReader()
docs = llama_reader.load_data("docs/EMPLOYEE_PRIVACY_POLICY_MINDERA_PT (2).pdf")
print(docs[0].text[:100] + "...")

#Split the text into chunks
from llama_index.core.node_parser import SentenceSplitter
text_splitter = SentenceSplitter(chunk_size=250, chunk_overlap=10)
#Create the index

from llama_index.core import VectorStoreIndex
index = VectorStoreIndex.from_documents(
    embed_model=embed_model,
    documents=docs,
    transformations=[text_splitter],
    show_progress=True
)


# 'similarity_top_k' refers to the number of top k chunks with the highest similarity.
k=5
#just to test the retriever
base_retriever = index.as_retriever(
    similarity_top_k=k)

# the real query engine for llm
query_engine = index.as_query_engine(
    llm=llm,
    streaming=True, similarity_top_k=k,
    verbose=True)

while True:
    query = input("Enter your query (or type 'exit' to quit): ")
    if query.lower() == 'exit':
        break

    source_nodes = base_retriever.retrieve(query)
#
    for node in source_nodes:
        print(f"---------------------------------------------")
        print(f"Score: {node.score:.3f}")
        print(node.get_content())
        print(f"---------------------------------------------\n\n")



    response = query_engine.query(query)
    response.print_response_stream()
    print(f"\n\n---------------------------------------------\n\n")