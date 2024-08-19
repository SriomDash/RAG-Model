import gradio as gr
import os

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

# Set the environment variable for HuggingFace API key
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "INSERT_YOUR_TOKEN"

chain = None  # Define chain as a global variable

def load_doc(file_path):
    global chain  # Access the global chain variable
    if file_path is None:
        return "Please upload a file"
    
    # Handle PDF files
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext != '.pdf':
        return "Unsupported file type. Please upload a PDF file."
    
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    embedding = HuggingFaceEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text = text_splitter.split_documents(documents)
    
    # Create the FAISS vector store
    db = FAISS.from_documents(text, embedding)
    
    # Initialize the language model with parameters explicitly set
    llm = HuggingFaceEndpoint(
        repo_id="OpenAssistant/oasst-sft-1-pythia-12b",
        temperature=1.0,
        max_length=256,
        huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"]
    )
    
    # Create the retrieval chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever()
    )
    
    return 'Document has successfully been loaded'

def answer_query(query_text):
    global chain  # Access the global chain variable
    if chain is None:
        return "Please load a PDF file first"
    return chain.run(query_text)

html = """
<div class="container">
    <h1>ChatPDF</h1>
    <p>Upload a PDF File, then click on Load PDF File <br>
    Once the document has been loaded you can begin chatting with the PDF =)</p>
</div>"""

css = """
.container {
    max-width: 700px;
    margin: 0 auto;
    padding: 20px;
    text-align: center;
}
"""

with gr.Blocks(css=css, theme=gr.themes.Monochrome()) as demo:
    gr.HTML(html)
    with gr.Column():
        gr.Markdown('ChatPDF')
        pdf_doc = gr.File(label="Load a File", file_types=['.pdf'], type='filepath')
        with gr.Row():
            load_pdf = gr.Button('Load File')
            status = gr.Textbox(label="Status", placeholder='', interactive=False)

        with gr.Row():
            input = gr.Textbox(label="Type in your question")
            output = gr.Textbox(label="Output")
        submit_query = gr.Button("Submit")

        load_pdf.click(load_doc, inputs=pdf_doc, outputs=status)
        submit_query.click(answer_query, inputs=input, outputs=output)

demo.launch()
