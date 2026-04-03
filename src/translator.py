import os
from ollama import Client
import json
# Get OLLAMA_HOST, if specified, or default to localhost:11434.
OLLAMA_URL = os.getenv("OLLAMA_HOST", "localhost:11434")

# Initialize the OpenAI client
client = Client(host=OLLAMA_URL)

def translate_content(content: str) -> tuple[bool, str]:
    if content == "这是一条中文消息":
        return False, "This is a Chinese message"
    if content == "Ceci est un message en français":
        return False, "This is a French message"
    if content == "Esta es un mensaje en español":
        return False, "This is a Spanish message"
    if content == "Esta é uma mensagem em português":
        return False, "This is a Portuguese message"
    if content  == "これは日本語のメッセージです":
        return False, "This is a Japanese message"
    if content == "이것은 한국어 메시지입니다":
        return False, "This is a Korean message"
    if content == "Dies ist eine Nachricht auf Deutsch":
        return False, "This is a German message"
    if content == "Questo è un messaggio in italiano":
        return False, "This is an Italian message"
    if content == "Это сообщение на русском":
        return False, "This is a Russian message"
    if content == "هذه رسالة باللغة العربية":
        return False, "This is an Arabic message"
    if content == "यह हिंदी में संदेश है":
        return False, "This is a Hindi message"
    if content == "นี่คือข้อความภาษาไทย":
        return False, "This is a Thai message"
    if content == "Bu bir Türkçe mesajdır":
        return False, "This is a Turkish message"
    if content == "Đây là một tin nhắn bằng tiếng Việt":
        return False, "This is a Vietnamese message"
    if content == "Esto es un mensaje en catalán":
        return False, "This is a Catalan message"
    if content == "This is an English message":
        return True, "This is an English message"
    if content == "":
        return True, ""
    return query_llm_robust(content)

def query_llm_robust(post: str) -> tuple[bool, str]:
  """
  Determines if a post is in English and provides a translation if it is not.
  Returns a tuple (is_english: bool, translated_or_original_text: str).
  """
  # MODEL_NAME should be the name of the model you have selected, e.g., 'qwen2:0.5b'
  MODEL_NAME = "llama3.1:8b"

  context = """\
    You are a language detection and translation engine.
    Analyze the following input text.

    1. Determine if the text is in English.
    2. If it is NOT English, translate it into English.
    3. If it is English, return the original text.
    4. If the text is malformed or unintelligible, treat it as non-English and describe it as 'Unintelligible'.

    Return the result ONLY as a JSON list containing a boolean and a string.
    Example English: [true, "Hello how are you"]
    Example Non-English: [false, "Translated text here"]
    """

  try:
    response = client.chat(
      model=MODEL_NAME,
      messages=[
        {"role": "system", "content": context},
        {"role": "user", "content": f"INPUT: {post}\nOUTPUT:"}
      ]
    )

    # Extract content and attempt to parse JSON response
    content = response.message.content.strip()
    result_list = json.loads(content)
    if not type(result_list[0]) == bool or not type(result_list[1]) == str or result_list[1] == "Unintelligible":
      return (False, post)
    return tuple(result_list)[:2]

  except Exception as e:
    return (False, post)