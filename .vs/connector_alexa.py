import logging
import json
from sanic import Blueprint, response
from sanic.request import Request
from typing import Text, Optional, List, Dict, Any

from rasa.core.channels.channel import UserMessage, OutputChannel
from rasa.core.channels.channel import InputChannel
from rasa.core.channels.channel import CollectingOutputChannel

logger = logging.getLogger(__name__)


class AlexaConnector(InputChannel):
    @classmethod
    def name(cls):
        return "dialect_quiz"

    def blueprint(self, on_new_message):

        alexa_webhook = Blueprint("alexa_webhook", __name__)

        @alexa_webhook.route("/", methods=["GET"])
        async def health(request):
            return response.json({"status": "ok"})

        @alexa_webhook.route("/webhook", methods=["POST"])
        async def receive(request):
            payload = request.json
            intenttype = payload["request"]["type"]
            if intenttype == "LaunchRequest":
                message = "Hello! Welcome to the Rasa-powered Alexa skill. You can start by saying hi."
                session = "false"
            else:
                intent = payload["request"]["intent"]["name"]
                print(intent)
                if intent == "AMAZON.StopIntent":
                    session = "true"
                    message = "Talk to you later"
                else:
                    text = payload["request"]["intent"]["slots"]["text"]["value"]
                    # print(text)
                    out = CollectingOutputChannel()
                    await on_new_message(UserMessage(text, out))
                    responses = [m["text"] for m in out.messages]
                    message = responses[0]
                    session = "false"
            r = {
                "version": "0.1",
                "sessionAttributes": {"status": "test"},
                "response": {
                    "outputSpeech": {
                        "type": "PlainText",
                        "text": message,
                        "playBehavior": "REPLACE_ENQUEUED",
                    },
                    "reprompt": {
                        "outputSpeech": {
                            "type": "PlainText",
                            "text": message,
                            "playBehavior": "REPLACE_ENQUEUED",
                        }
                    },
                    "shouldEndSession": session,
                },
            }

            return response.json(r)

        return alexa_webhook
