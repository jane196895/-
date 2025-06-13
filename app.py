import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from typing import List
import os
import fitz  # PyMuPDF
import re

# 환경 변수 로드
from dotenv import load_dotenv
load_dotenv()

# OpenAI API 키 불러오기
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY가 설정되지 않았습니다. Streamlit Cloud에서는 Secrets 설정을 확인하세요.")
    st.stop()

############################### 1단계 : PDF 문서를 벡터DB에 저장 ##########################

def save_uploadedfile(uploadedfile: UploadedFile) -> str:
    temp_dir = "PDF_임시폴더"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, uploadedfile.name)
    with open(file_path, "wb") as f:
        f.write(uploadedfile.read())
    return file_path

def pdf_to_documents(pdf_path: str) -> List[Document]:
    loader = PyMuPDFLoader(pdf_path)
    doc = loader.load()
    for d in doc:
        d.metadata['file_path'] = pdf_path
    return doc

def chunk_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return splitter.split_documents(documents)

def save_to_vector_store(documents: List[Document]) -> None:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
    vector_store = FAISS.from_documents(documents, embedding=embeddings)
    vector_store.save_local("faiss_index")

############################### 2단계 : 질문 처리 및 RAG ##########################

@st.cache_data
def process_question(user_question):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = new_db.as_retriever(search_kwargs={"k": 3})
    retrieve_docs = retriever.invoke(user_question)
    chain = get_rag_chain()
    response = chain.invoke({"question": user_question, "context": retrieve_docs})
    return response, retrieve_docs

def get_rag_chain() -> Runnable:
    template = """
    다음의 컨텍스트를 활용해서 질문에 답변해줘
    - 상황
    - 교사의 권장반응
    - 생활지도 방법
    - 법적근거
    컨텍스트 : {context}
    
    질문: {question}
    
    응답:"""
    prompt = PromptTemplate.from_template(template)
    model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)
    return prompt | model | StrOutputParser()

############################### 3단계 : PDF 이미지 처리 ##########################

@st.cache_data(show_spinner=False)
def convert_pdf_to_images(pdf_path: str, dpi: int = 250) -> List[str]:
    doc = fitz.open(pdf_path)
    image_paths = []
    output_folder = "PDF_이미지"
    os.makedirs(output_folder, exist_ok=True)

    for i, page in enumerate(doc):
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        image_path = os.path.join(output_folder, f"page_{i + 1}.png")
        pix.save(image_path)
        image_paths.append(image_path)
    return image_paths

def display_pdf_page(image_path: str, page_number: int):
    with open(image_path, "rb") as f:
        st.image(f.read(), caption=f"Page {page_number}", output_format="PNG", width=600)

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]

############################### main 함수 ##########################

def main():
    st.set_page_config("학생생활지도 FAQ 챗봇", layout="wide")
    left_column, right_column = st.columns([1, 1])

    with left_column:
        st.header("학생생활지도 FAQ 챗봇")

        pdf_doc = st.file_uploader("PDF Uploader", type="pdf")
        if st.button("PDF 업로드하기") and pdf_doc:
            with st.spinner("PDF문서 저장중"):
                pdf_path = save_uploadedfile(pdf_doc)
                docs = pdf_to_documents(pdf_path)
                chunks = chunk_documents(docs)
                save_to_vector_store(chunks)
            with st.spinner("PDF 페이지를 이미지로 변환중"):
                images = convert_pdf_to_images(pdf_path)
                st.session_state.images = images

        user_question = st.text_input("PDF 문서에 대해 질문해 주세요",
                                      placeholder="예: 수업시간에 돌아다니는 학생이 있을 때 어떻게 지도하나요?")

        if user_question:
            response, context = process_question(user_question)
            st.write(response)
            for i, doc in enumerate(context):
                with st.expander("관련 문서"):
                    st.write(doc.page_content)
                    file_path = doc.metadata.get('file_path', '')
                    page_number = doc.metadata.get('page', 0) + 1
                    button_key = f"link_{i}_{page_number}"
                    if st.button(f"🔎 문서 p.{page_number}", key=button_key):
                        st.session_state.page_number = page_number

    with right_column:
        page_number = st.session_state.get('page_number')
        if page_number:
            image_folder = "PDF_이미지"  # ✅ 대소문자 정확히 일치
            if not os.path.exists(image_folder):
                st.warning(f"이미지 폴더 '{image_folder}'가 존재하지 않습니다. PDF를 먼저 업로드하세요.")
                return
            image_files = sorted(os.listdir(image_folder), key=natural_sort_key)
            image_paths = [os.path.join(image_folder, fname) for fname in image_files]
            if 0 < page_number <= len(image_paths):
                display_pdf_page(image_paths[page_number - 1], page_number)
            else:
                st.warning("해당 페이지 이미지를 찾을 수 없습니다.")

if __name__ == "__main__":
    main()
    
    