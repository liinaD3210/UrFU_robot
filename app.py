import streamlit as st
import os
import re
import docx2txt
from typing import List, TypedDict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from langchain_core.pydantic_v1 import BaseModel, Field

# --- 1. ЗАГРУЗКА И НАСТРОЙКА ---

# Загружаем переменные окружения (GOOGLE_API_KEY)
load_dotenv()

# Настройка страницы Streamlit
st.set_page_config(page_title="Гид по ИРИТ-РТФ", page_icon="🤖", layout="wide")
st.title("Агент-консультант для абитуриентов ИРИТ-РТФ")
st.caption("Я отвечаю на вопросы только на основе загруженной базы знаний.")


# --- 2. ПАРСИНГ ДАННЫХ И СОЗДАНИЕ RETRIEVER ---

@st.cache_resource(show_spinner="Загружаю и индексирую базу знаний...")
def load_and_index_data(file_path: str):
    """
    Загружает DOCX файл, парсит его на отдельные направления подготовки
    и создает векторный retriever (поисковик).
    """
    if not os.path.exists(file_path):
        st.error(f"Файл не найден по пути: {file_path}")
        st.stop()

    text = docx2txt.process(file_path)
    # Используем regex для разделения текста на блоки по коду направления (e.g., 09.03.01)
    # re.S (dotall) позволяет '.' совпадать с переносом строки
    chunks = re.findall(r'(\d{2}\.\d{2}\.\d{2}.*?)(?=\d{2}\.\d{2}\.\d{2}|\Z)', text, re.S)

    documents = []
    for chunk in chunks:
        # Извлекаем название направления (первая строка) для метаданных
        title = chunk.split('\n', 1)[0].strip()
        doc = Document(page_content=chunk, metadata={"source": title})
        documents.append(doc)

    if not documents:
        st.error("Не удалось распарсить документ на направления. Проверьте формат файла.")
        st.stop()

    # Создаем эмбеддинги с помощью модели Google
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Создаем векторную базу FAISS для быстрого поиска
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Создаем retriever, который будет возвращать до 3 самых похожих документов
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return retriever, documents

# Запускаем загрузку данных. Результат кэшируется.
retriever, all_documents = load_and_index_data("data/irit_rtf_baccalaureate_info.docx")



# --- 3. ОПРЕДЕЛЕНИЕ ГРАФА ДИАЛОГА (LANGGRAPH) ---

class GraphState(TypedDict):
    """Определяет состояние нашего графа"""
    original_question: str # Новый ключ для хранения исходного вопроса
    question: str          # Текущий рабочий вопрос
    documents: List[Document] # Найденные документы
    generation: str        # Сгенерированный LLM ответ
    clarification_needed: bool # Флаг, нужно ли уточнение

# Инициализируем LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, convert_system_message_to_human=True)

# --- Узлы графа ---

# --- Узлы графа ---

def retrieve_docs(state: GraphState) -> GraphState:
    """
    Узел для поиска релевантных документов.
    ЭТОТ УЗЕЛ МОДИФИЦИРОВАН для решения проблемы бесконечного уточнения.
    """
    print("--- УЗЕЛ: ПОИСК ДОКУМЕНТОВ (с логикой уточнения) ---")
    question = state["question"]
    
    # --- НАЧАЛО НОВОЙ ЛОГИКИ ---
    # Сначала проверяем, не является ли вопрос точным выбором из предложенных вариантов.
    # Мы ищем точное совпадение названия направления (из метаданных) в тексте вопроса.
    # Это срабатывает, когда пользователь нажимает на кнопку с названием.
    for doc in all_documents:
        # doc.metadata['source'] содержит полное название, например "11.03.01 Радиотехника"
        doc_title = doc.metadata.get("source", "")
        if doc_title and doc_title.lower() in question.lower():
            print(f"--- НАЙДЕНО ТОЧНОЕ СОВПАДЕНИЕ ПО НАЗВАНИЮ: {doc_title} ---")
            # Если нашли, то не используем векторный поиск, а сразу возвращаем этот один документ.
            # Это и есть ключ к разрыву цикла уточнений.
            documents = [doc]
            # Используем правильный порядок обновления стейта
            return {
                "documents": documents,
                "question": question,
                "generation": "",
                "clarification_needed": False
            }
    # --- КОНЕЦ НОВОЙ ЛОГИКИ ---

    # Если точного совпадения не найдено, это первичный запрос.
    # Выполняем обычный векторный поиск.
    print("--- Точного совпадения нет, выполняю векторный поиск по всему тексту ---")
    documents = retriever.invoke(question)
    return {
        "documents": documents,
        "question": question,
        "original_question": question,
        "generation": "",
        "clarification_needed": False
    }

from langchain_core.pydantic_v1 import BaseModel, Field

# Структура для оценки релевантности
class RelevanceGrade(BaseModel):
    """Бинарная оценка релевантности документа вопросу."""
    score: str = Field(description="Отвечает ли документ на вопрос, 'yes' или 'no'.")

# LLM с настроенным выводом для оценки
structured_llm_grader = llm.with_structured_output(RelevanceGrade)

# Промпт для грейдера
grader_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Ты — эксперт по оценке релевантности. Твоя задача - проверить, содержит ли предоставленный документ ответ на вопрос пользователя. Отвечай только 'yes' или 'no'."),
        ("human", "Документ:\n\n{document}\n\nВопрос: {question}"),
    ]
)

relevance_grader = grader_prompt | structured_llm_grader

def grade_documents(state: GraphState) -> GraphState:
    """
    Узел для оценки найденных документов.
    - Отбрасывает нерелевантные.
    - Решает, нужно ли уточнение, если релевантных > 1.
    """
    print("--- УЗЕЛ: УМНАЯ ОЦЕНКА ДОКУМЕНТОВ ---")
    question = state["question"]
    documents = state["documents"]
    
    if not documents:
        return {**state, "documents": []}

    # Оцениваем каждый документ
    filtered_docs = []
    for d in documents:
        grade = relevance_grader.invoke({"question": question, "document": d.page_content})
        if grade.score.lower() == "yes":
            print(f"--- ДОКУМЕНТ '{d.metadata.get('source', '')}' РЕЛЕВАНТЕН ---")
            filtered_docs.append(d)
        else:
            print(f"--- ДОКУМЕНТ '{d.metadata.get('source', '')}' НЕРЕЛЕВАНТЕН ---")
    
    # Решаем, нужно ли уточнение
    # Если на вопрос "где есть физика?" нашлось 3 релевантных документа,
    # то уточнять НЕ нужно, нужно делать сводный ответ.
    # Уточнение нужно, если вопрос был нечеткий, например "расскажи про IT".
    # Для простоты, пока будем считать, что если нашлось несколько релевантных, то делаем сводку.
    clarification_needed = False # По умолчанию, делаем сводный ответ
    if len(filtered_docs) > 1:
        # Здесь можно добавить более сложную логику, например,
        # еще один вызов LLM, чтобы спросить "нужно ли уточнение для этого вопроса?"
        # Но для начала, упростим: если вопрос про конкретные атрибуты (физика, баллы),
        # то уточнение не нужно. Если общий - нужно.
        # Для нашего кейса "куда с физикой" - clarification_needed = False
        print("--- Найдено несколько релевантных документов, будет сгенерирован сводный ответ. ---")

    return {**state, "documents": filtered_docs, "clarification_needed": clarification_needed}


def generate_answer(state: GraphState) -> GraphState:
    """Узел для генерации ответа с помощью LLM, ИСПОЛЬЗУЕТ ORIGINAL_QUESTION."""
    print("--- УЗЕЛ: ГЕНЕРАЦИЯ ОТВЕТА ---")
    # Используем ИСХОДНЫЙ вопрос для сохранения контекста!
    question = state["original_question"]
    documents = state["documents"]

    prompt_template = ChatPromptTemplate.from_template(
        """Ты — чат-бот-помощник абитуриента ИРИТ-РТФ.
Твоя задача — отвечать на вопросы, основываясь ИСКЛЮЧИТЕЛЬНО на предоставленном ниже контексте.
Синтезируй информацию из всех предоставленных фрагментов, чтобы дать полный и исчерпывающий ответ на исходный вопрос пользователя.
Если контекст позволяет, сначала дай краткий сводный ответ (например, перечисли направления), а затем опиши детали.

КОНТЕКСТ:
{context}

ИСХОДНЫЙ ВОПРОС:
{question}
"""
    )
    
    rag_chain = prompt_template | llm | StrOutputParser()
    context_str = "\n\n---\n\n".join([doc.page_content for doc in documents])
    generation = rag_chain.invoke({"context": context_str, "question": question})
    
    return {**state, "generation": generation}

def generate_clarification(state: GraphState) -> GraphState:
    """
    Узел для формирования уточняющего вопроса.
    ВОЗВРАЩАЕТ ТОЛЬКО МАШИНОЧИТАЕМЫЙ МАРКЕР И СПИСОК.
    """
    print("--- УЗЕЛ: ФОРМИРОВАНИЕ УТОЧНЕНИЯ ---")
    documents = state["documents"]
    doc_titles = [doc.metadata.get("source", "Неизвестное направление") for doc in documents]
    
    # НОВЫЙ ФОРМАТ: только маркер и опции, разделенные переносом строки.
    # Интерфейс сам добавит вводный текст.
    clarification_message = "CLARIFY_OPTIONS:\n" + "\n".join(doc_titles)
    
    return {**state, "generation": clarification_message}


def fallback(state: GraphState) -> GraphState:
    """Узел для ответа, если ничего не найдено."""
    print("--- УЗЕЛ: ОТВЕТ-ЗАГЛУШКА ---")
    generation = "К сожалению, я не нашел информации по вашему запросу в своей базе знаний. Попробуйте переформулировать вопрос."
    # ПРАВИЛЬНЫЙ ПОРЯДОК: сначала старый стейт, потом новое значение
    return {**state, "generation": generation}


# --- Условные переходы графа ---

def decide_next_step(state: GraphState) -> str:
    """Определяет следующий шаг на основе оценки документов."""
    print("--- УСЛОВИЕ: ПРИНЯТИЕ РЕШЕНИЯ ---")
    if not state["documents"]:
        print("--- РЕШЕНИЕ: Ничего не найдено -> fallback ---")
        return "fallback"
    if state["clarification_needed"]:
        print("--- РЕШЕНИЕ: Нужно уточнение -> clarify ---")
        return "clarify"
    else:
        print("--- РЕШЕНИЕ: Всё ясно -> generate ---")
        return "generate"


# --- Сборка графа ---
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve_docs)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate_answer)
workflow.add_node("clarify", generate_clarification)
workflow.add_node("fallback", fallback)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_next_step,
    {
        "clarify": "clarify",
        "generate": "generate",
        "fallback": "fallback",
    },
)
workflow.add_edge("generate", END)
workflow.add_edge("clarify", END)
workflow.add_edge("fallback", END)

# Компилируем граф в исполняемый объект
app = workflow.compile()


# --- 4. ИНТЕРФЕЙС STREAMLIT (ФИНАЛЬНАЯ ВЕРСИЯ С ДЕАКТИВАЦИЕЙ КНОПОК) ---

# Инициализация истории чата
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Здравствуйте! Чем я могу вам помочь с выбором направления в ИРИТ-РТФ?"}]

# Коллбэк для кнопок уточнения. Он только сохраняет выбор.
def handle_clarification_click(option_text):
    st.session_state.clarification_choice = option_text

# 1. БЛОК ОТРИСОВКИ: всегда рисуем из истории
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        content = message["content"]
        # Проверяем, есть ли в сообщении ассистента маркер для кнопок
        if isinstance(content, str) and content.startswith("CLARIFY_OPTIONS:"):
            st.markdown("Я нашел несколько подходящих направлений. Пожалуйста, уточните, какое из них вас интересует:")
            options = content.split('\n')[1:]
            num_columns = min(len(options), 3)
            cols = st.columns(num_columns)
            for i, option in enumerate(options):
                if option.strip():
                    with cols[i % num_columns]:
                        st.button(
                            option,
                            on_click=handle_clarification_click,
                            args=[option],
                            use_container_width=True,
                            # Ключ теперь может быть и не уникальным в истории, т.к. мы убираем старые кнопки
                            key=f"clarify_btn_{option.replace(' ', '_')}_{i}" 
                        )
        else:
            # Если это обычное сообщение, просто рисуем его
            st.markdown(content)

# 2. БЛОК ОБРАБОТКИ ВВОДА

# Сначала проверяем, не был ли сделан выбор кнопкой (это имеет приоритет)
if prompt_from_button := st.session_state.get("clarification_choice"):
    # Сбрасываем триггер
    st.session_state.clarification_choice = None
    
    # --- КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: ДЕАКТИВАЦИЯ СТАРЫХ КНОПОК ---
    # Находим последнее сообщение с кнопками и заменяем его на текст выбора
    for i in range(len(st.session_state.messages) - 1, -1, -1):
        msg = st.session_state.messages[i]
        if msg["role"] == "assistant" and msg["content"].startswith("CLARIFY_OPTIONS:"):
            msg["content"] = f"Вы уточнили свой выбор: **{prompt_from_button}**"
            break
    # --- КОНЕЦ ИЗМЕНЕНИЯ ---
    
    # Добавляем "виртуальный" ввод пользователя в историю
    st.session_state.messages.append({"role": "user", "content": f"Расскажи подробнее про \"{prompt_from_button}\""})
    # Принудительно перезапускаем скрипт, чтобы запустить обработку
    st.rerun()

# Если не было нажатия на кнопку, проверяем ручной ввод
if user_input := st.chat_input("Задайте ваш вопрос...", key="main_chat_input"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.rerun()

# 3. БЛОК ГЕНЕРАЦИИ ОТВЕТА (остается без изменений)
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    user_question = st.session_state.messages[-1]["content"]
    with st.chat_message("assistant"):
        with st.spinner("Думаю..."):
            final_state = app.invoke({"question": user_question})
            response = final_state['generation']
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()