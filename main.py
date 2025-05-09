import os
import io
import logging
import asyncio
import traceback

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command

import torch
from transformers import pipeline, WhisperForConditionalGeneration, WhisperProcessor
import soundfile as sf

# Чтение переменных окружения
BOT_TOKEN = os.getenv('BOT_TOKEN')
ALLOWED_CHAT_ID = [int(x) for x in os.getenv('ALLOWED_CHAT_ID').split(',')]
PYTORCH_DEVICE = os.getenv('PYTORCH_DEVICE')
SPEECH_RECOGNITION_MODEL = os.getenv('SPEECH_RECOGNITION_MODEL')

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация бота
bot = Bot(token=BOT_TOKEN)

# Инициализация диспетчера
dp = Dispatcher()

# Настройка модели распознавания речи
logger.info("Настройка модели распознавания речи...")
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
processor = WhisperProcessor.from_pretrained(SPEECH_RECOGNITION_MODEL)
model = WhisperForConditionalGeneration.from_pretrained(SPEECH_RECOGNITION_MODEL)

# Установка языка и задачи для принудительного декодирования (только русский)
language = "ru"
task = "transcribe"
forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
model.generation_config.forced_decoder_ids = forced_decoder_ids

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=30,  # Позволяет обрабатывать аудио длиннее 30 секунд
    device=PYTORCH_DEVICE,
    torch_dtype=torch_dtype
)

# Установка pad_token_id
if pipe.tokenizer.pad_token_id is None or pipe.tokenizer.pad_token_id == pipe.tokenizer.eos_token_id:
    pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id + 1
logger.info("Модель распознавания речи успешно настроена.")

@dp.message(Command("start"))
async def start_message(message: types.Message):
    if message.chat.id not in ALLOWED_CHAT_ID:
        await message.reply(f"Этот бот недоступен в этом чате. ID чата {message.chat.id}")
        return
    await message.reply(
        'Добро пожаловать! Этот бот может распознать ваш *голос* в голосовом сообщении и преобразовать '
        'его в *текст*.\nОтправьте голосовое сообщение, чтобы начать конвертацию.',
        parse_mode='Markdown'
    )

@dp.message(lambda message: message.voice)
async def media_handler(message: types.Message):
    if message.chat.id not in ALLOWED_CHAT_ID:
        await message.reply(f"Этот бот недоступен в этом чате. ID чата {message.chat.id}")
        return

    try:
        if message.voice:
            file_id = message.voice.file_id
            file_size = message.voice.file_size
        else:
            await message.reply('Неподдерживаемый тип медиа.')
            return

        file = await bot.get_file(file_id)
        file_path = file.file_path
        audio_stream = await bot.download_file(file_path)

        text = await voice_recognizer(audio_stream)
        await message.reply(text)
    except Exception as e:
        logger.error(f"Ошибка в media_handler:\n{traceback.format_exc()}")
        await message.reply('Произошла ошибка при обработке вашего сообщения.')

@dp.message(Command("get_chat_id"))
async def get_chat_id(message: types.Message):
    chat_id = message.chat.id
    await message.reply(f"ID этого чата: {chat_id}")

async def voice_recognizer(audio_stream):
    converted_audio = await convert_audio(audio_stream)
    if not converted_audio:
        return "Не удалось преобразовать аудио."

    try:
        # Загрузка аудио данных из объекта BytesIO
        converted_audio.seek(0)
        data, samplerate = sf.read(converted_audio)

        # Передача аудиоданных напрямую в pipe без преобразования в тензор
        transcription = pipe(data, chunk_length_s=30)["text"]
        return transcription.strip()
    except Exception:
        logger.error(f"Ошибка при транскрипции:\n{traceback.format_exc()}")
        return "Не удалось транскрибировать аудио."

async def convert_audio(audio_data):
    ffmpeg_cmd = [
        'ffmpeg',
        '-i', 'pipe:0',               # Вход из stdin
        '-vn',                        # Игнорировать видео (если это видеофайл)
        '-f', 'wav',                  # Формат вывода
        '-ar', '16000',               # Изменение частоты дискретизации на 16000 Гц
        '-ac', '1',                   # Убедиться, что канал только один (моно)
        'pipe:1',                     # Вывод в stdout
        '-y',                         # Перезапись выходных файлов
        '-loglevel', 'error'          # Подавление ненужного вывода
    ]
    try:
        process = await asyncio.create_subprocess_exec(
            *ffmpeg_cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        out, err = await process.communicate(input=audio_data.getvalue())
        if process.returncode != 0:
            logger.error(f"Ошибка FFmpeg: {err.decode()}")
            return None
        else:
            return io.BytesIO(out)
    except Exception as e:
        logger.error(f"Ошибка в convert_audio:\n{traceback.format_exc()}")
        return None

async def main():
    logger.info('Запуск бота...')
    await dp.start_polling(bot)
    logger.info('Бот остановлен.')

if __name__ == '__main__':
    asyncio.run(main())
