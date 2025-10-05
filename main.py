import os
import io
import logging
import asyncio
import time
import traceback

from aiogram import Bot, Dispatcher, types, enums
from aiogram.filters import Command

import torch
from transformers import pipeline, WhisperForConditionalGeneration, WhisperProcessor
import soundfile as sf

# Чтение переменных окружения
BOT_TOKEN = os.environ['BOT_TOKEN']
ALLOWED_CHAT_ID = [int(x) for x in os.environ['ALLOWED_CHAT_ID'].split(',')]
PYTORCH_DEVICE = os.getenv('PYTORCH_DEVICE')
SPEECH_RECOGNITION_MODEL = os.environ['SPEECH_RECOGNITION_MODEL']

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация бота
bot = Bot(token=BOT_TOKEN)

# Инициализация диспетчера
dp = Dispatcher()

# Настройка модели распознавания речи
logger.info("Настройка модели распознавания речи...")
torch_dtype = torch.float16 if torch.cuda.is_available() and PYTORCH_DEVICE != 'cpu' else torch.float32
processor = WhisperProcessor.from_pretrained(SPEECH_RECOGNITION_MODEL)
model = WhisperForConditionalGeneration.from_pretrained(SPEECH_RECOGNITION_MODEL)

# Установка языка и задачи для принудительного декодирования (только русский)
model.generation_config.language = 'ru'
model.generation_config.task = 'transcribe'

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=PYTORCH_DEVICE,
    torch_dtype=torch_dtype,
    return_timestamps=True
)

# Установка pad_token_id
if pipe.tokenizer.pad_token_id is None or pipe.tokenizer.pad_token_id == pipe.tokenizer.eos_token_id:
    pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id + 1
logger.info("Модель распознавания речи успешно настроена.")

@dp.message(Command("start"))
async def start_message(message: types.Message) -> None:
    if message.chat.id not in ALLOWED_CHAT_ID:
        await message.reply(f"Этот бот недоступен в этом чате. ID чата {message.chat.id}")
        return
    await message.reply(
        'Добро пожаловать! Этот бот может распознать ваш *голос* в голосовом сообщении и преобразовать '
        'его в *текст*.\nОтправьте голосовое сообщение, чтобы начать конвертацию.',
        parse_mode='Markdown'
    )

@dp.message(lambda message: message.voice or message.video_note)
async def media_handler(message: types.Message) -> None:
    if message.chat.id not in ALLOWED_CHAT_ID:
        await message.reply(f"Этот бот недоступен в этом чате. ID чата {message.chat.id}")
        return

    try:
        if message.voice:
            file_id = message.voice.file_id
            tmp_path = os.path.join('/app/tmp', f'{time.time_ns()}.ogg')
        elif message.video_note:
            file_id = message.video_note.file_id
            tmp_path = os.path.join('/app/tmp', f'{time.time_ns()}.mp4')
        else:
            return
        
        file = await bot.get_file(file_id)
        file_path = file.file_path
        if not file_path:
            return
        await bot.download_file(file_path, destination=tmp_path)
            
        # Передаем тип файла в конвертер
        text = await voice_recognizer(tmp_path)
        await message.reply(f"<blockquote expandable>{text}</blockquote>", parse_mode=enums.parse_mode.ParseMode.HTML)
        os.remove(tmp_path)
    except Exception as e:
        logger.error(f"Ошибка в media_handler:\n{e}\n{traceback.format_exc()}")
        await message.reply('Произошла ошибка при обработке вашего сообщения.')

@dp.message(Command("get_chat_id"))
async def get_chat_id(message: types.Message) -> None:
    chat_id = message.chat.id
    await message.reply(f"ID этого чата: {chat_id}")

async def voice_recognizer(audio_path: str) -> str:
    # Передаем тип файла в конвертер
    converted_audio = await convert_audio(audio_path)
    if converted_audio is None:
        return "Не удалось преобразовать аудио."

    try:
        converted_audio.seek(0)
        data, samplerate = sf.read(converted_audio)
        transcription = pipe(data)["text"]
        return str(transcription.strip())
    except Exception:
        logger.error(f"Ошибка при транскрипции:\n{traceback.format_exc()}")
        return "Не удалось транскрибировать аудио."


async def convert_audio(audio_path: str) -> io.BytesIO | None:    
    # Определяем формат входного файла
    ffmpeg_cmd = [
        'ffmpeg',
        '-i', audio_path,         # Вход из stdin
        '-vn',                    # Игнорировать видео
        '-f', 'wav',              # Формат вывода
        '-ar', '16000',           # Частота дискретизации
        '-ac', '1',               # Моно-аудио
        'pipe:1',                 # Вывод в stdout
        '-y',                     # Перезапись выходных файлов
        '-loglevel', 'error'      # Подавление ненужного вывода
    ]
    
    try:
        process = await asyncio.create_subprocess_exec(
            *ffmpeg_cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        # Асинхронная запись и чтение
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            logger.error(f"FFmpeg error: {stderr.decode()}")
            return None
            
        return io.BytesIO(stdout)
    except Exception as e:
        logger.error(f"Ошибка в convert_audio: {traceback.format_exc()}")
        return None

async def main() -> None:
    logger.info('Запуск бота...')
    await dp.start_polling(bot)
    logger.info('Бот остановлен.')

if __name__ == '__main__':
    asyncio.run(main())
