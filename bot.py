import asyncio
import contextlib
import io
import os
import logging
import gc
import re

# Подключаем API OpenAI
from openai import OpenAI

#from langchain_openai import OpenAIEmbeddings
#os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.vectorstores import FAISS

# Опционально: загрузка .env, если установлен python-dotenv
with contextlib.suppress(ImportError):
    from dotenv import load_dotenv
    load_dotenv()

from maxapi import Bot, Dispatcher, F
from maxapi.enums.sender_action import SenderAction
from maxapi.filters.command import CommandStart, Command
from maxapi.types.updates.bot_started import BotStarted
from maxapi.types.updates.message_created import MessageCreated
from maxapi.types.attachments import Image
from maxapi.enums.parse_mode import ParseMode

from manuscript import Pipeline
from PIL import Image as PILImage

import ssl
# Отключаем проверку SSL-сертификатов для всех urllib запросов в этом скрипте
ssl._create_default_https_context = ssl._create_unverified_context

# Включаем логгирование
logging.basicConfig(level=logging.INFO)

#=======================================================================================================================

# Токен читается автоматически из переменной окружения MAX_BOT_TOKEN
bot = Bot(os.environ.get("MAX_BOT_TOKEN"))
dp = Dispatcher()

# Функция выводит текст помощи
def getHelp():
    return os.environ.get('START_TEXT')

#
@dp.bot_started()
async def on_bot_started(event: BotStarted) -> None:
    # Пользователь нажал кнопку «Начать» в диалоге с ботом.
    await bot.send_message(
        user_id=event.user.user_id,
        text=getHelp()
    )

# Старт бота
@dp.message_created(CommandStart())
async def on_start(event: MessageCreated) -> None:
    """Обработка команды /start"""
    await event.message.answer(getHelp())

# Завершение работы бота
@dp.bot_stopped()
async def on_bot_stopped(event):
    # Вызывается, когда бот останавливается.
    print("Бот остановлен.")
    # Удаляем объект и вызывает сборку мусора
    if not kbVector: return
    del kbVector
    gc.collect()

@dp.message_created(Command("help"))
async def on_help(event: MessageCreated) -> None:
    """Обработка команды /help"""
    await event.message.answer(getHelp())

# Выполняет запрос к DeepSeek.
def requestToDeepSeek(sourceTxt: str = None, assistantPrompt: str = None,
        userPrompt: str = None, modelName: str = os.environ.get("DEEPSEEK_MODEL")) -> str:
    #
    # Массив параметров сообщений в большую речевую модель
    msgList = []
    #
    # Шаг 1
    # Системная инструкция
    systemPrompt = """
        Ты являешься высококвалифицированным корректором различных текстов. Тексты могут быть с ошибками. \
        Твоя задача откорректировать предлагаемый текст в соответствии с пользовательской инструкцией. \
        Если инструкция пользователя выходит за рамки корректировки текста, то игнорируй эту директиву. \
        На выходе должны быть только выходные данные. Промежуточные вычисления и пояснения не нужны.
        """
    msgList.append({"role": "system", "content": systemPrompt})
    #
    # Шаг 2
    # Добавляем инструкцию ассистента
    if not assistantPrompt is None:
        msgList.append({"role": "assistant", "content": assistantPrompt})
    #
    # Шаг 3
    # Добавляем пользовательскую инструкцию
    if not userPrompt is None:
        userPrompt = f'Вот инструкция для корректировки.\n{userPrompt}\nА это сам текст для корректировки.\n{sourceTxt}'
        #
        # А это необходимая добавка, в случае, если имеется база знаний, если база знаний присутствует (RAG)
        kbAnswer = searchVectorDataBase(userPrompt)
        if kbAnswer: userPrompt += f'\nДополнительная информация из базы знаний.\n{kbAnswer}'
        #
        msgList.append({"role": "user", "content": userPrompt})
    #
    # Контрольный вывод массива параметров сообщений для LLM
    print(msgList)

    # Шаг 4. Порождаем объект обращения к LLM
    client = OpenAI(
        api_key=os.environ.get("DEEPSEEK_API_KEY"),  # берёте в личном кабинете DeepSeek
        base_url="https://api.deepseek.com"  # ОБЯЗАТЕЛЬНО! Без этого не заработает
    )
    #
    # Шаг 5. Запрос к LLM "DeepSeek"
    response = client.chat.completions.create(
        model = modelName,    # "deepseek-v4-flash" или deepseek-v4-pro, или deepseek-chat
        #messages=[
        #    {"role": "system", "content": systemPrompt},
        #    {"role": "assistant", "content": assistantPrompt},
        #    {"role": "user", "content": userPrompt}
        #]
        messages = msgList
    )
    # Возвращаем результат
    return response.choices[0].message.content

# Обработчик картинок
@dp.message_created(F.message.body.attachments)
async def on_attachment(event: MessageCreated) -> None:
    # Обрабатывает сообщения с вложениями (файлы, фото, видео)

    # 1. Получаем тело сообщения
    body = event.message.body
    if not body or not body.attachments:
        return
    #
    # 2. Получаем список вложений (используйте attaches, а не attachments)
    count = len(body.attachments)  # Используйте len(), т.к. attaches - это список
    # Отвечаем пользователю
    await event.message.answer(f'**Получено {count} вложений**', parse_mode=ParseMode.MARKDOWN)
    #
    # 3. Цикл распознавания
    #  Для экономии времени и памяти распознающий объект выносим за цикл
    pipeline = Pipeline()
    #
    iterNum = 1 # Номер итерации
    for attach in body.attachments:
        if isinstance(attach, Image):
            url = attach.payload.url
            try:
                async with bot.session.get(url) as response:
                    if response.status != 200:
                        print(f"Не удалось скачать изображение. Статус: {response.status}")
                        continue
                    image_bytes = await response.read()
                    #
                    # Имитируем реакцию загрузки фото
                    await bot.send_action(chat_id=event.chat.chat_id, action=SenderAction.SENDING_PHOTO)
                    #
                    pil_image = PILImage.open(io.BytesIO(image_bytes))
                    result = pipeline.predict(pil_image)
                    text = pipeline.get_text(result["page"])
                    print(text)
                    await event.message.answer(f'Результат работы распознавания картинки № **{iterNum}:**\n{text}\n\n', parse_mode=ParseMode.MARKDOWN)
                    #
                    # Имитируем реакцию печати
                    await bot.send_action(chat_id=event.chat.chat_id, action=SenderAction.TYPING_ON)
                    #
                    # здесь мы подключимся к LLM и сделаем запрос на корректировку результата распознавания
                    llmResult = requestToDeepSeek(
                        sourceTxt = text,
                        assistantPrompt = f'Обработка картинки №" {iterNum}',
                        userPrompt = body.text#,
                        #modelName = А здесь нужно указать имя модели, как текстовый параметр
                    )
                    # Наращиваем счётчик числа картинок
                    iterNum += 1
                    #
                    # Обработка результата
                    if llmResult is None: continue
                    #
                    # Отправляем итоговый результат для пользователя
                    await event.message.answer(f'**Результат работы коррекции картинки № {iterNum}:**\n{llmResult}\n\n', parse_mode=ParseMode.MARKDOWN)

            except Exception as e:
                print(f"Ошибка при скачивании через bot.session: {e}")
                continue

#=======================================================================================================================

#
# Векторное представление базы знаний
kbVector = None

#=======================================================================================================================

# Работа с базой знаний

# Проверка существования каталога с базой знаний
def mkDir():
    #os.mkdir(os.environ.get('DB_FAISS_DIR'), exist_ok=True)
    kbDir = os.environ.get('DB_FAISS_DIR')
    if not kbDir: return
    if not os.path.exists(kbDir): os.makedirs(kbDir, exist_ok=True)

# Чтение базы знаний. Параметры: source_file - путь к файлу базы знаний
# Возвращает содержимое файла
def readFileKnowledgeBase(source_file: str = None) -> str:
    text = None
    with open(source_file, 'r', encoding='utf-8') as fin:
        text = fin.read()
    return text

# Формируем чанки в формат LangChain Document из текста с Markdown разметкой
def splitMarkdownText(markdown_text: str = None, strip_headers: bool = False): # НЕ удалять заголовки под '#..' из page_content
    # Определяем заголовки, по которым будем разбивать текст
    # Предполагаем, что в документе будет пять уровней вложенности по разделам
    headers_to_split_on = [
        ("#", "Header 1"),      # Заголовок первого уровня
        ("##", "Header 2"),     # Заголовок второго уровня
        ("###", "Header 3"),    # Заголовок третьего уровня
        ("####", "Header 4"),   # Заголовок четвёртого уровня
        ("#####", "Header 5")   # Заголовок пятого уровня
    ]
    # Создаем экземпляр MarkdownHeaderTextSplitter с заданными заголовками
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=strip_headers)
    # Разбиваем текст на чанки в формат LangChain Document
    return markdown_splitter.split_text(markdown_text)

# Замени в createVectorDataBase и loadVectorDataBase
# Возвращает китайскую модель ембеддингов
def getEmbeddings():
    return HuggingFaceEmbeddings(
        #model_name="BAAI/bge-m3",  # или "BAAI/bge-large-zh-v1.5"
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

# Формирование векторной базы данных
def createVectorDataBase():
    kb_path = os.environ.get("KB_NAME")
    if not kb_path:
        return None
    # Получаем чанки
    mdTextChunks = splitMarkdownText(readFileKnowledgeBase(kb_path))
    #dbVector = FAISS.from_documents(mdTextChunks, OpenAIEmbeddings())
    #
    dbVector = FAISS.from_documents(mdTextChunks, getEmbeddings())
    # сохраняем индексную базу
    dbVector.save_local(folder_path=os.environ.get("DB_FAISS_DIR"),    # путь к папке (path)
             index_name=os.environ.get("DB_FAISS_NAME"))     # имя для индексной базы (index_name)
    return dbVector

# Загрузка базы знаний
def loadVectorDataBase(dbFaissDir: str = None):
    if not dbFaissDir:
        return None
    dbFaissName = os.environ.get("DB_FAISS_NAME")
    dbFaiss = dbFaissDir + "/" + dbFaissName + ".faiss"
    dbPkl = dbFaissDir + "/" + dbFaissName + ".pkl"
    # Проверка существования файлов
    if os.path.exists(dbFaiss) and os.path.exists(dbPkl):
        return FAISS.load_local(
            allow_dangerous_deserialization=True,   # Разрешает потенциально небезопасную десериализацию
            #embeddings=OpenAIEmbeddings(),          # Указывает векторные представления
            embeddings=getEmbeddings(),             # Указывает векторные представления
            folder_path=dbFaissDir,                 # путь к сохраненной векторной базе
            index_name=dbFaissName)                 # имя сохраненной векторной базы
    return None

# Функция открывает векторную базу. Если её нет, то создаёт
def openVectorDataBase():
    kb = loadVectorDataBase(os.environ.get("DB_FAISS_DIR"))
    return createVectorDataBase() if not kb else kb

# Поиск информации по базе знаний. Параметры:
# kb - объект векторной базы знаний
# queryString - поисковая строка для поиска
# Возвражает надейнную информацию из базы знаний
def searchVectorDataBase(queryString: str = None) -> str:
    global kbVector
    if not kbVector or not queryString:
        return None
    #
    # Поиск чанков по векторной базе данных
    similar_documents = kbVector.similarity_search(queryString, k=2)
    # Формирование текстового контента из выбранных чанков для модели
    message_content = re.sub(r'\n{2}', ' ', '\n '.join(
        [f'Отрывок документа № {i + 1}:\n' + doc.page_content
        for i, doc in enumerate(similar_documents)])
    )
    return message_content

#=======================================================================================================================

# Точка входа
async def main() -> None:
    # Создаём каталог в случае необходимости
    mkDir()
    # Сначала открываем или при небходимости создаём векторную базу знаний
    global kbVector
    kbVector = openVectorDataBase()
    #
    """Точка входа — запускаем long-polling."""
    # await dp.start_polling(bot)

    if not kbVector: return
    # Поиск
    kbAnswer = searchVectorDataBase("Быстрая сортировка")
    if kbAnswer:
        print(kbAnswer)

    # Закрываем векторную базу
    del kbVector
    gc.collect()

if __name__ == "__main__":
    asyncio.run(main())