"""Gemini-backed responses for the MoneyMinds WhatsApp agent."""

import logging
import os
import threading

from google import genai
from google.genai import types


# Tried in order; the first model that answers wins. The newest Flash models
# regularly return 503 on the free tier, so always keep a fallback behind them.
MODEL_CHAIN = [
    "gemini-flash-lite-latest",
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
]

# How many user+model exchanges to keep per conversation.
MAX_HISTORY_TURNS = 10

# Sent when every model in the chain fails, so the user never gets silence.
FALLBACK_MESSAGE = (
    "Sorry, I can't reach my assistant right now. Please try again in a moment.\n"
    "Desole, mo pa kapav reponn la. Reseye dan enn ti moman."
)

SYSTEM_PROMPT = """You are MoneyMinds, a financial assistant for Mauritius that \
talks to people over WhatsApp.

LANGUAGE
Reply in the same language the user wrote to you in: English, French, or Kreol \
Morisien. If they mix languages, follow their lead. Never apologise for the \
language you were addressed in.

WHAT YOU DO
1. Scam checks. When someone forwards a suspicious SMS or message, say plainly \
whether it is a scam. Then point to the specific things in THAT message that give \
it away - the exact domain used, the urgency, the request for a PIN - so the person \
can spot the next one themselves. Finish with what to do now.
2. Budgeting. Help people build a spending plan around their real income, track it, \
and notice overspending early.
3. Investing literacy. Explain Mauritian options in plain words: SEM-listed shares, \
MCB and SBM funds, government savings bonds, platforms like Fundkiss. Explain how \
they work and how to open an account.

HARD RULES
- Never state a phone number, URL, email address, or branch address unless it appears \
in VERIFIED CONTACTS below. If you do not have it, tell the person to use the number \
printed on their bank card or the bank's official app. A wrong number here is worse \
than no number.
- Never ask for a PIN, password, OTP, card number, or account number. Say plainly \
that you will never ask for them.
- Teach, do not advise. Explain how instruments work and what the trade-offs are. \
Do not tell anyone what to buy, how much to invest, or how to allocate their money. \
You are not a licensed financial adviser; say so if pushed.
- If you are not sure about a fact, say you are not sure.

STYLE
WhatsApp messages, not essays. Under 120 words unless asked for more. Short lines, \
no headers. Use *single asterisks* for emphasis, sparingly.

VERIFIED CONTACTS
(none configured yet - do not state any contact details)
"""


_client = None
_client_lock = threading.Lock()

_histories = {}
_history_lock = threading.Lock()


def _get_client():
    """Build the Gemini client once, lazily, so import never needs the key."""
    global _client
    with _client_lock:
        if _client is None:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY is not set")
            _client = genai.Client(api_key=api_key)
        return _client


def _history_for(wa_id):
    with _history_lock:
        return list(_histories.get(wa_id, []))


def _remember(wa_id, role, text):
    with _history_lock:
        history = _histories.setdefault(wa_id, [])
        history.append({"role": role, "parts": [{"text": text}]})
        # Keep only the most recent turns so the prompt stays small.
        del history[: -MAX_HISTORY_TURNS * 2]


def reset_conversation(wa_id):
    """Forget a conversation, e.g. when the user asks to start over."""
    with _history_lock:
        _histories.pop(wa_id, None)


def generate_response(message_body, wa_id, name=None):
    """Ask Gemini for a reply, walking the model chain until one answers."""
    contents = _history_for(wa_id)
    contents.append({"role": "user", "parts": [{"text": message_body}]})

    system_prompt = SYSTEM_PROMPT
    if name:
        system_prompt += f"\nThe person you are talking to is called {name}."

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=0.3,
        max_output_tokens=800,
    )

    last_error = None
    for model in MODEL_CHAIN:
        try:
            response = _get_client().models.generate_content(
                model=model, contents=contents, config=config
            )
        except Exception as e:  # 503s and quota errors are expected on free tier
            last_error = e
            logging.warning(f"Gemini model {model} failed: {e}")
            continue

        text = (response.text or "").strip()
        if not text:
            logging.warning(f"Gemini model {model} returned no text")
            continue

        _remember(wa_id, "user", message_body)
        _remember(wa_id, "model", text)
        logging.info(f"Gemini answered via {model}")
        return text

    logging.error(f"All Gemini models failed. Last error: {last_error}")
    return FALLBACK_MESSAGE
