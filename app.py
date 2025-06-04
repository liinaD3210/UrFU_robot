# app.py
import streamlit as st
import os
import json
from dotenv import load_dotenv
import re
from typing import List, Dict # Добавил Dict для типизации

# Langchain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever # Изменено здесь

# Загрузка переменных окружения (API ключа)
load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    st.error("Не найден GOOGLE_API_KEY. Пожалуйста, добавьте его в файл .env")
    st.stop()

JSON_DATA_PATH = "data/programs_data.json" 

@st.cache_resource
def load_programs_data() -> tuple[List[dict], Dict[str, dict]]: # Добавил тип возвращаемого значения
    if not os.path.exists(JSON_DATA_PATH):
        st.error(f"Файл с данными программ не найден: {JSON_DATA_PATH}")
        st.stop()
    with open(JSON_DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    programs_search_dict: Dict[str, dict] = {} # Явно типизируем
    for program in data:
        if "название_программы" in program and program["название_программы"]:
            # Добавляем название программы и каждый профиль как отдельные ключи
            base_name_lower = program["название_программы"].lower()
            programs_search_dict[base_name_lower] = program
            
            # Если есть профиль, добавляем его тоже как ключ для поиска
            # (возможно, объединенный с базовым названием)
            if program.get("профиль_или_специализация"):
                profile_name_lower = program["профиль_или_специализация"].lower()
                programs_search_dict[profile_name_lower] = program # Поиск только по названию профиля
                programs_search_dict[f"{base_name_lower} {profile_name_lower}"] = program # Поиск по "название + профиль"

        if "id" in program and program["id"]: # Если есть код направления
            programs_search_dict[program["id"]] = program
        
        # Дополнительно: можно разбить название на слова и добавить их как ключи,
        # но это может привести к слишком многим совпадениям.
        # words_in_name = re.findall(r'\b\w{3,}\b', program.get("название_программы", "").lower())
        # for word in words_in_name:
        #     if word not in programs_search_dict: # Чтобы не перезаписывать более точные совпадения
        #         programs_search_dict[word] = program


    st.info(f"Загружено {len(data)} программ из JSON. Создан поисковый словарь с {len(programs_search_dict)} ключами.")
    return data, programs_search_dict

class JsonProgramRetriever(BaseRetriever):
    programs_list: List[dict]
    programs_search_dict: dict # Словарь для быстрого поиска по ключам
    k: int = 1 

    def _get_relevant_documents(self, query: str, *, run_manager = None) -> List[Document]:
        relevant_programs_data: List[dict] = []
        query_lower = query.lower().strip()

        # 0. Сначала ищем по ID, если он есть в запросе - это самый точный поиск
        match_id = re.search(r'\b(\d{2}\.\d{2}\.\d{2}(?:-\w+)?)\b', query_lower) # Ищем код типа XX.YY.ZZ или XX.YY.ZZ-суффикс
        if match_id:
            program_id_from_query = match_id.group(1)
            if program_id_from_query in self.programs_search_dict:
                prog_data = self.programs_search_dict[program_id_from_query]
                if prog_data not in relevant_programs_data:
                    relevant_programs_data.append(prog_data)
        
        # 1. Попытка точного/частичного совпадения по ключам в search_dict
        # (ключи: полные названия, ID, профили)
        if len(relevant_programs_data) < self.k:
            # Сортируем ключи словаря по длине в обратном порядке, чтобы сначала проверять более длинные (специфичные) ключи
            sorted_search_keys = sorted(self.programs_search_dict.keys(), key=len, reverse=True)
            for key_search in sorted_search_keys:
                if key_search in query_lower: 
                    program_data_val = self.programs_search_dict[key_search]
                    if program_data_val not in relevant_programs_data: 
                        relevant_programs_data.append(program_data_val)
                    if len(relevant_programs_data) >= self.k*2: # Собираем чуть больше, потом отсортируем по релевантности
                        break 
        
        # 2. Если совпадений по ключам мало, пробуем более гибкий поиск по отдельным словам в названии/профиле
        if len(relevant_programs_data) < self.k:
            query_words = set(re.findall(r'\b\w{3,}\b', query_lower)) # Слова от 3 букв
            
            candidate_programs = []
            for program_data_val in self.programs_list:
                if program_data_val in relevant_programs_data: continue

                # Собираем текст для поиска: название + профиль + описание (первые N слов)
                text_to_search_in = program_data_val.get("название_программы", "").lower()
                if program_data_val.get("профиль_или_специализация"):
                    text_to_search_in += " " + program_data_val["профиль_или_специализация"].lower()
                
                # Можно добавить начало описания для более широкого поиска
                # description_preview = " ".join(program_data_val.get("описание", "").lower().split()[:20]) # первые 20 слов
                # text_to_search_in += " " + description_preview

                # Считаем количество совпавших слов
                matched_words_count = len(query_words.intersection(set(re.findall(r'\b\w+\b', text_to_search_in))))

                if matched_words_count > 0:
                    candidate_programs.append({"program": program_data_val, "matches": matched_words_count})
            
            # Сортируем кандидатов по количеству совпавших слов
            if candidate_programs:
                sorted_candidates = sorted(candidate_programs, key=lambda x: x["matches"], reverse=True)
                for cand in sorted_candidates:
                    if cand["program"] not in relevant_programs_data:
                        relevant_programs_data.append(cand["program"])
                    if len(relevant_programs_data) >= self.k*2: # Собираем чуть больше
                        break
        
        # Удаляем дубликаты, сохраняя порядок (сначала более релевантные)
        final_relevant_programs = []
        seen_ids = set()
        for prog in relevant_programs_data:
            prog_id_key = prog.get("id", str(prog)) # Уникальный ключ для программы
            if prog_id_key not in seen_ids:
                final_relevant_programs.append(prog)
                seen_ids.add(prog_id_key)

        if not final_relevant_programs:
            st.warning(f"Отладка ретривера: Не удалось найти программу по запросу: '{query}'")
            return [Document(page_content="Информация по вашему запросу о программе не найдена в базе данных.")]

        # Форматируем найденные программы в документы
        docs = []
        st.info(f"Отладка ретривера: Найдено {len(final_relevant_programs)} программ-кандидатов для запроса '{query}'. Берем топ-{self.k}.")
        for i, program_data_item in enumerate(final_relevant_programs[:self.k]): 
            content_str = f"Информация по программе \"{program_data_item.get('название_программы', 'Неизвестно')}\":\n"
            if program_data_item.get('профиль_или_специализация'):
                content_str += f"- Профиль/Специализация: {program_data_item['профиль_или_специализация']}\n"
            
            # Отображаем поля в определенном порядке для лучшей читаемости
            display_order = ["id", "срок_обучения_лет", "обязательные_предметы", 
                             "предметы_по_выбору_примечание", "предметы_по_выбору_детали",
                             "проходной_балл_2024", "бюджетные_места", "описание"]
            
            processed_keys = set(["название_программы", "профиль_или_специализация"])

            for key_item in display_order:
                if key_item in program_data_item and program_data_item[key_item] is not None:
                    value_item = program_data_item[key_item]
                    formatted_key = key_item.replace('_', ' ').capitalize()
                    
                    if key_item == "описание" and isinstance(value_item, str) and len(value_item) > 300: # Сокращаем длинное описание
                         content_str += f"- {formatted_key}: {value_item[:300]}...\n"
                    elif isinstance(value_item, list) and value_item: # Для списков (например, предметы)
                        content_str += f"- {formatted_key}:\n"
                        for item_list_el in value_item:
                            if isinstance(item_list_el, dict): # Если элемент списка - словарь
                                item_str_dict = ", ".join([f"{ik_dict.replace('_', ' ').capitalize() if ik_dict != 'название' else ''}{': ' if ik_dict != 'название' else ''}{iv_dict}" for ik_dict, iv_dict in item_list_el.items()])
                                content_str += f"  - {item_str_dict}\n"
                            else: # Если элемент списка - простое значение
                                content_str += f"  - {item_list_el}\n"
                    else: # Для обычных значений
                        content_str += f"- {formatted_key}: {value_item}\n"
                    processed_keys.add(key_item)

            # Добавляем остальные поля, если они есть и не были обработаны
            for key_item, value_item in program_data_item.items():
                if key_item not in processed_keys and value_item is not None:
                    formatted_key = key_item.replace('_', ' ').capitalize()
                    content_str += f"- {formatted_key}: {value_item}\n"

            docs.append(Document(page_content=content_str, metadata={"source": "json_database", "program_name": program_data_item.get("название_программы")}))
            if i == 0: # Печатаем только первый найденный документ для отладки
                 st.info(f"Отладка ретривера: Первый найденный документ для LLM:\n{content_str[:500]}...")
        
        return docs

    async def _aget_relevant_documents(self, query: str, *, run_manager = None) -> List[Document]:
        # Для простоты, используем синхронную версию. Для продакшена лучше реализовать асинхронно.
        return self._get_relevant_documents(query, run_manager=run_manager)


@st.cache_resource
def setup_llm_and_retriever():
    programs_list, programs_search_dict = load_programs_data()
    
    custom_retriever = JsonProgramRetriever(
        programs_list=programs_list,
        programs_search_dict=programs_search_dict,
        k=1 
    )

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.2, # Еще ниже для строгих ответов по JSON
            convert_system_message_to_human=True
        )
    except Exception as e:
        st.error(f"Ошибка при инициализации LLM Google: {e}")
        st.stop()

    prompt_template_str = """Ты — ИИ-ассистент, отвечающий на вопросы об образовательных программах ИРИТ-РТФ.
Используй ТОЛЬКО предоставленную информацию о программе (контекст), чтобы ответить на вопрос.
Если в контексте нет ответа на вопрос, скажи: "В предоставленной информации о программе нет ответа на этот вопрос." или "Я не нашел эту информацию в описании программы."
Не придумывай информацию. Отвечай на русском языке.

Контекст (информация о программе):
{context}

Вопрос: {question}
Ответ на русском языке на основе контекста:"""
    PROMPT = PromptTemplate(
        template=prompt_template_str, input_variables=["context", "question"]
    )

    chain_type_kwargs = {"prompt": PROMPT}
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=custom_retriever, 
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs
    )
    return qa_chain

# --- Streamlit UI ---
st.set_page_config(page_title="ИРИТ-РТФ Гид (JSON)", layout="centered")
st.title("🤖 Гид по ИРИТ-РТФ (на основе JSON)")

try:
    qa_chain = setup_llm_and_retriever()
except Exception as e:
    st.error(f"Критическая ошибка при настройке приложения: {e}")
    st.stop() 

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Здравствуйте! Задайте вопрос о направлении подготовки ИРИТ-РТФ."}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ваш вопрос (например, 'Расскажи про Радиотехнику' или 'Какие предметы на 09.03.01?')..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        with st.spinner("Ищу информацию и думаю..."):
            try:
                result = qa_chain({"query": prompt})
                answer = result.get("result", "Не удалось получить ответ.")
                source_documents = result.get("source_documents")
                
                full_response = answer
                
                if source_documents and not (len(source_documents) == 1 and "Информация по вашему запросу о программе не найдена в базе данных." in source_documents[0].page_content):
                    full_response += "\n\n<details><summary>Источник информации (данные по программе):</summary>\n"
                    for doc in source_documents:
                        content_html = doc.page_content.replace('\n', '<br>') 
                        full_response += f"<p><small><i>{content_html}</i></small></p>\n"
                    full_response += "</details>"

            except Exception as e:
                st.error(f"Ошибка при обработке запроса: {e}")
                full_response = f"Произошла ошибка: {e}"
        
        message_placeholder.markdown(full_response, unsafe_allow_html=True)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})

if len(st.session_state.messages) > 1 :
    if st.button("🗑️ Очистить чат"):
        st.session_state.messages = [{"role": "assistant", "content": "Здравствуйте! Задайте вопрос о направлении подготовки ИРИТ-РТФ."}]
        st.rerun()