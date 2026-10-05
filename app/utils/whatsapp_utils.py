import logging
from flask import current_app, jsonify
import json
import requests
import re
import threading
from collections import OrderedDict

from app.services.gemini_service import generate_response


def log_http_response(response):
    logging.info(f"Status: {response.status_code}")
    logging.info(f"Content-type: {response.headers.get('content-type')}")
    logging.info(f"Body: {response.text}")


def get_text_message_input(recipient, text):
    return json.dumps(
        {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": recipient,
            "type": "text",
            "text": {"preview_url": False, "body": text},
        }
    )


# Meta retries webhooks it considers failed, so the same message can be
# delivered more than once. Remember the ids we have already answered.
MAX_SEEN_MESSAGE_IDS = 1000
_seen_message_ids = OrderedDict()
_seen_lock = threading.Lock()


def already_handled(message_id):
    """True if this message id has been seen before (and record it if not)."""
    if not message_id:
        return False
    with _seen_lock:
        if message_id in _seen_message_ids:
            return True
        _seen_message_ids[message_id] = None
        while len(_seen_message_ids) > MAX_SEEN_MESSAGE_IDS:
            _seen_message_ids.popitem(last=False)
        return False


def send_message(data):
    headers = {
        "Content-type": "application/json",
        "Authorization": f"Bearer {current_app.config['ACCESS_TOKEN']}",
    }

    url = f"https://graph.facebook.com/{current_app.config['VERSION']}/{current_app.config['PHONE_NUMBER_ID']}/messages"

    try:
        response = requests.post(
            url, data=data, headers=headers, timeout=10
        )  # 10 seconds timeout as an example
        response.raise_for_status()  # Raises an HTTPError if the HTTP request returned an unsuccessful status code
    except requests.Timeout:
        logging.error("Timeout occurred while sending message")
        return jsonify({"status": "error", "message": "Request timed out"}), 408
    except (
        requests.RequestException
    ) as e:  # This will catch any general request exception
        logging.error(f"Request failed due to: {e}")
        return jsonify({"status": "error", "message": "Failed to send message"}), 500
    else:
        # Process the response as normal
        log_http_response(response)
        return response


def process_text_for_whatsapp(text):
    # Remove brackets
    pattern = r"\【.*?\】"
    # Substitute the pattern with an empty string
    text = re.sub(pattern, "", text).strip()

    # Pattern to find double asterisks including the word(s) in between
    pattern = r"\*\*(.*?)\*\*"

    # Replacement pattern with single asterisks
    replacement = r"*\1*"

    # Substitute occurrences of the pattern with the replacement
    whatsapp_style_text = re.sub(pattern, replacement, text)

    return whatsapp_style_text


def process_whatsapp_message(body):
    value = body["entry"][0]["changes"][0]["value"]
    wa_id = value["contacts"][0]["wa_id"]
    name = value["contacts"][0]["profile"]["name"]

    message = value["messages"][0]
    message_type = message.get("type")

    if already_handled(message.get("id")):
        logging.info(f"Skipping repeat delivery of message {message.get('id')}")
        return

    # Only text messages carry a body we can read. Images, audio, reactions and
    # the like would raise a KeyError below and return a 500 to Meta, which
    # retries and can eventually disable the webhook.
    if message_type != "text":
        logging.info(f"Ignoring unsupported message type '{message_type}' from {wa_id}")
        return

    message_body = message["text"]["body"]

    response = generate_response(message_body, wa_id, name)
    # Gemini writes **bold**; WhatsApp expects *bold*.
    response = process_text_for_whatsapp(response)

    # Reply to whoever sent the message, not a hardcoded recipient
    data = get_text_message_input(wa_id, response)
    send_message(data)


def is_valid_whatsapp_message(body):
    """
    Check if the incoming webhook event has a valid WhatsApp message structure.
    """
    return (
        body.get("object")
        and body.get("entry")
        and body["entry"][0].get("changes")
        and body["entry"][0]["changes"][0].get("value")
        and body["entry"][0]["changes"][0]["value"].get("messages")
        and body["entry"][0]["changes"][0]["value"]["messages"][0]
    )
