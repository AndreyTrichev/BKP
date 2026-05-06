import logging

from aiogram import Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
from aiogram.enums import ChatAction

from bot.client import MLClient

logger = logging.getLogger(__name__)
router = Router()
ml_client = MLClient()


DEPT_NAMES = {
    "urban_economy": "Управление городского хозяйства (УГХ)",
    "urban_development": "Управление городского развития (УГР)",
    "education": "Управление образования",
    "culture": "Управление культуры",
    "unknown": "не определено",
}

SUBDEPT_NAMES = {
    "gas": "Газоснабжение",
    "water": "Водоснабжение",
    "heat": "Теплоснабжение",
    "housing": "Содержание МКД",
    "roads": "Дороги и тротуары",
    "waste": "Вывоз мусора",
    "ecology": "Экология",
    "construction": "Строительство",
    "land": "Земельные участки",
    "planning": "Генплан и НТО",
    "trade": "Торговля",
    "beaches": "Пляжи",
    "tourism": "Туризм",
    "preschool": "Детские сады",
    "school": "Школы",
    "custody": "Опека",
    "institutions": "Учреждения культуры",
    "heritage": "Культурное наследие",
}


WELCOME = """Здравствуйте! Я помогу направить ваше обращение в нужный отдел Администрации г. Феодосия.

Просто опишите вашу проблему — я определю, к какому отделу она относится, и оператор свяжется с вами.

Например: «Батареи холодные третий день», «Не убирают мусор во дворе», «Когда зачислят ребёнка в детский сад».
"""


@router.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer(WELCOME)


@router.message(Command("help"))
async def cmd_help(message: Message):
    await message.answer(WELCOME)


@router.message()
async def handle_appeal(message: Message):
    text = (message.text or "").strip()
    if not text:
        await message.answer("Пожалуйста, опишите вашу проблему текстом.")
        return

    if len(text) < 5:
        await message.answer(
            "Пожалуйста, опишите ситуацию подробнее — хотя бы пару предложений."
        )
        return

                                                     
    await message.bot.send_chat_action(
        chat_id=message.chat.id, action=ChatAction.TYPING
    )

    try:
        result = await ml_client.classify(text)
    except Exception as e:
        logger.exception("classify failed")
        await message.answer(
            "Извините, сервис временно недоступен. Попробуйте через минуту."
        )
        return

    dept = result["department"]
    sub = result["subdepartment"]
    confidence = result["confidence"]
    fallback = result["fallback"]
    fallback_reason = result.get("fallback_reason")
    raw_score = result.get("raw_score", 0.0)

                                                                         
    if fallback:
        if fallback_reason == "raw_floor" and raw_score < 0.001:
                                                                                 
            await message.answer(
                "❌ Похоже, ваше обращение не относится к компетенции "
                "Администрации г. Феодосия.\n\n"
                "Если это ошибка — попробуйте описать ситуацию подробнее. "
                "Если нужна помощь по другой теме — обратитесь в "
                "соответствующее ведомство (Роспотребнадзор, полиция, прокуратура и т.п.)."
            )
        elif fallback_reason == "raw_floor":
                                                                  
            await message.answer(
                "⚠️ Не уверены, что обращение относится к компетенции Администрации.\n\n"
                "Передаём оператору на проверку — он рассмотрит и свяжется с вами."
            )
        elif fallback_reason == "gap_min":
                                                                  
            await message.answer(
                "🤔 В обращении затронуто несколько тем — мы не смогли точно "
                "определить основной отдел.\n\n"
                "Передаём оператору, он уточнит и свяжется с вами."
            )
        else:
                                                                    
            await message.answer(
                "Ваше обращение пока не удалось точно определить.\n"
                "Оператор Администрации рассмотрит его и свяжется с вами."
            )
        return

                                                                          
    dept_name = DEPT_NAMES.get(dept, dept)
    sub_name = SUBDEPT_NAMES.get(sub or "", sub or "")

    if confidence == "high":
        prefix = "✅ Обращение направлено:"
    elif confidence == "medium":
        prefix = "🟡 Обращение направлено (предварительно):"
    else:
        prefix = "🔄 Обращение направлено на проверку оператору:"

    response = f"{prefix}\n\n📂 {dept_name}"
    if sub_name:
        response += f"\n📁 {sub_name}"
    response += "\n\nОператор отдела свяжется с вами."

    await message.answer(response)
