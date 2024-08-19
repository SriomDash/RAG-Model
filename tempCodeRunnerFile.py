import gradio as gr
import os
import win32com.client  # For PPT to PDF conversion on Windows
import subprocess        # For PPT to PDF conversion using LibreOffice

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

# Set the environment variable for HuggingFace API key
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

chain = None  # Define chain as a global variable

def convert_ppt_to_pdf(ppt_file, pdf_file):
    if os.name == 'nt':  # Windows
        # Create a PowerPoint application object
        powerpoint = win32com.client.Dispatch("PowerPoint.Application")
        # Open the PowerPoint file
        presentation = powerpoint.Presentations.Open(ppt_file)
        # Save as PDF
        presentation.SaveAs(pdf_file, FileFormat=32)  # 32 corresponds to PDF format
        # Close the presentation
        presentation.Close()
        # Quit PowerPoint
        powerpoint.Quit()
    else:
        # Use LibreOffice for cross-platform support
        command = [
            'libreoffice', '--headless', '--convert-to', 'pdf',
            '--outdir', os.path.dirname(pdf_file), ppt_file
        ]
        subprocess.run(command, check=True)

def load_doc(file_path):
    global chain  # Access the global chain variable
    if file_path is None:
        return "Please upload a file"
    
    # Determine file type and convert to PDF if needed
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext in ['.ppt', '.pptx']:
        pdf_path = os.path.splitext(file_path)[0] + '.pdf'
        convert_ppt_to_pdf(file_path, pdf_path)
        file_path = pdf_path

    # Handle PDF files
    if file_ext != '.pdf':
        return "Unsupported file type. Please upload a PDF or PPT file."
    
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
<div style="text-align:center; max-width: 700px;">
    <h1>ChatPDF</h1>
    <p> Upload a PDF File, then click on Load PDF File <br>
    Once the document has been loaded you can begin chatting with the PDF =)
</div>"""

css = """
.container {
    max-width: 700px;
    margin-left: auto;
    margin-right: auto;
    padding: 20px;
}
"""


with gr.Blocks(css=css, theme=gr.themes.Monochrome()) as demo:
    gr.HTML(html)
    with gr.Column():
        gr.Markdown('ChatPDF')
        pdf_doc = gr.File(label="Load a File", file_types=['.pdf', '.ppt', '.pptx'], type='filepath')
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
