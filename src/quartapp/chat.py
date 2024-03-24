import json
import os

import azure.identity.aio
import openai
from quart import (
    Blueprint,
    Response,
    current_app,
    render_template,
    request,
    stream_with_context,
)

bp = Blueprint("chat", __name__, template_folder="templates", static_folder="static")


@bp.before_app_serving
async def configure_openai():
    client_args = {}
    if os.getenv("LOCAL_OPENAI_ENDPOINT"):
        # Use a local endpoint like llamafile server
        current_app.logger.info("Using local OpenAI-compatible API with no key")
        client_args["api_key"] = "no-key-required"
        client_args["base_url"] = os.getenv("LOCAL_OPENAI_ENDPOINT")
        bp.openai_client = openai.AsyncOpenAI(
            **client_args,
        )
    else:
        # Use the OpenAI service with an API token
        if os.getenv("OPENAI_AUTH_TOKEN"):
            # Authenticate using an OpenAI API token
            current_app.logger.info("Using OpenAI service with API token")
            client_args["api_key"] = os.getenv("OPENAI_AUTH_TOKEN")
        else:
            current_app.logger.error("No OpenAI API token provided")
            return
        bp.openai_client = openai.AsyncOpenAI(
            **client_args,
        )


@bp.after_app_serving
async def shutdown_openai():
    await bp.openai_client.close()


@bp.get("/")
async def index():
    return await render_template("index.html")


@bp.post("/chat")
async def chat_handler():
    request_messages = (await request.get_json())["messages"]

    @stream_with_context
    async def response_stream():
        # This sends all messages, so API request may exceed token limits
        all_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
        ] + request_messages

        model_name = os.environ.get("AZURE_OPENAI_CHATGPT_DEPLOYMENT", "gpt-3.5-turbo")
        chat_coroutine = bp.openai_client.chat.completions.create(
            model=model_name,
            messages=all_messages,
            stream=True,
        )
        try:
            async for event in await chat_coroutine:
                current_app.logger.info(event)
                yield json.dumps(event.model_dump(), ensure_ascii=False) + "\n"
        except Exception as e:
            current_app.logger.error(e)
            yield json.dumps({"error": str(e)}, ensure_ascii=False) + "\n"

    return Response(response_stream())
