from src.translator import translate_content, client
from mock import patch
from sentence_transformers import SentenceTransformer, util
from typing import Callable

model = SentenceTransformer('all-MiniLM-L6-v2')


@patch.object(client, 'chat')
def test_unexpected_language(mocker):
    """
    Mock tests
    """
    # we mock the model's response to return a random message
    mocker.return_value.message.content = "I don't understand your request"
    assert translate_content("Hier ist dein erstes Beispiel.") == (False, "Hier ist dein erstes Beispiel.")

    # Empty string
    mocker.return_value.message.content = ""
    assert translate_content("Hier ist dein erstes Beispiel.") == (False, "Hier ist dein erstes Beispiel.")

    # Plain translated text with no JSON wrapper
    mocker.return_value.message.content = "Here is the first example."
    assert translate_content("Hier ist dein erstes Beispiel.") == (False, "Hier ist dein erstes Beispiel.")

    # --- Wrong JSON structure ---
    # Number instead of bool
    mocker.return_value.message.content = '[1, "Here is the first example."]'
    assert translate_content("Hier ist dein erstes Beispiel.") == (False, "Hier ist dein erstes Beispiel.")

    # String instead of bool
    mocker.return_value.message.content = '["false", "Here is the first example."]'
    assert translate_content("Hier ist dein erstes Beispiel.") == (False, "Hier ist dein erstes Beispiel.")

    # JSON object instead of list
    mocker.return_value.message.content = '{"is_english": false, "text": "Here is the first example."}'
    assert translate_content("Hier ist dein erstes Beispiel.") == (False, "Hier ist dein erstes Beispiel.")

    # List too short (missing translation string)
    mocker.return_value.message.content = '[false]'
    assert translate_content("Hier ist dein erstes Beispiel.") == (False, "Hier ist dein erstes Beispiel.")

    # Empty list
    mocker.return_value.message.content = '[]'
    assert translate_content("Hier ist dein erstes Beispiel.") == (False, "Hier ist dein erstes Beispiel.")

    # JSON null
    mocker.return_value.message.content = 'null'
    assert translate_content("Hier ist dein erstes Beispiel.") == (False, "Hier ist dein erstes Beispiel.")

    # List with extra elements — result must still be a valid (bool, str) tuple
    mocker.return_value.message.content = '[false, "Here is the first example.", "extra"]'
    result = translate_content("Hier ist dein erstes Beispiel.")
    assert isinstance(result, tuple) and len(result) == 2
    assert isinstance(result[0], bool) and isinstance(result[1], str)

    # Markdown-wrapped JSON (common LLM artifact)
    mocker.return_value.message.content = '```json\n[false, "Here is the first example."]\n```'
    assert translate_content("Hier ist dein erstes Beispiel.") == (False, "Hier ist dein erstes Beispiel.")

    # 'Unintelligible' sentinel value triggers fallback
    mocker.return_value.message.content = '[false, "Unintelligible"]'
    assert translate_content("asdfghjkl") == (False, "asdfghjkl")

    # --- Client raises an exception (e.g. network failure) ---
    mocker.side_effect = RuntimeError("Connection refused")
    assert translate_content("Hier ist dein erstes Beispiel.") == (False, "Hier ist dein erstes Beispiel.")
    mocker.side_effect = None  # reset for remaining assertions

    # Valid response
    mocker.return_value.message.content = '[false, "Here is the first example."]'
    assert translate_content("Hier ist dein erstes Beispiel.") == (False, "Here is the first example.")

    # Not valid, but should still pass
    mocker.return_value.message.content = '[true, "Here is the first example."]'
    assert translate_content("Hier ist dein erstes Beispiel.") == (True, "Here is the first example.")

    # Also not valid but should still pass
    mocker.return_value.message.content = '[false, "idk"]'
    assert translate_content("Hier ist dein erstes Beispiel.") == (False, "idk")

normal_eval_set = [
    # ── English posts (15) ── bool=True, string = original post as-is ──
    {
        "post": "Hello, how are you?",
        "expected_answer": (True, "Hello, how are you?")
    },
    {
        "post": "The weather is really nice today!",
        "expected_answer": (True, "The weather is really nice today!")
    },
    {
        "post": "I love programming and building cool projects.",
        "expected_answer": (True, "I love programming and building cool projects.")
    },
    {
        "post": "Can you help me with this assignment?",
        "expected_answer": (True, "Can you help me with this assignment?")
    },
    {
        "post": "The quick brown fox jumps over the lazy dog.",
        "expected_answer": (True, "The quick brown fox jumps over the lazy dog.")
    },
    {
        "post": "I just got a new puppy and he is absolutely adorable!",
        "expected_answer": (True, "I just got a new puppy and he is absolutely adorable!")
    },
    {
        "post": "Machine learning is transforming every industry.",
        "expected_answer": (True, "Machine learning is transforming every industry.")
    },
    {
        "post": "Please submit your homework before midnight.",
        "expected_answer": (True, "Please submit your homework before midnight.")
    },
    {
        "post": "What time does the meeting start tomorrow?",
        "expected_answer": (True, "What time does the meeting start tomorrow?")
    },
    {
        "post": "I can't believe how fast this semester is going.",
        "expected_answer": (True, "I can't believe how fast this semester is going.")
    },
    {
        "post": "Did you watch the game last night? It was incredible!",
        "expected_answer": (True, "Did you watch the game last night? It was incredible!")
    },
    {
        "post": "I need to pick up groceries on the way home.",
        "expected_answer": (True, "I need to pick up groceries on the way home.")
    },
    {
        "post": "The library closes at 9 PM on weekdays.",
        "expected_answer": (True, "The library closes at 9 PM on weekdays.")
    },
    {
        "post": "My favorite season is autumn because of the colors.",
        "expected_answer": (True, "My favorite season is autumn because of the colors.")
    },
    {
        "post": "Could you recommend a good restaurant nearby?",
        "expected_answer": (True, "Could you recommend a good restaurant nearby?")
    },

    # ── Non-English posts (15) ── bool=False, string = English translation ──
    {
        "post": "Hier ist dein erstes Beispiel.",
        "expected_answer": (False, "This is your first example.")  # German
    },
    {
        "post": "Bonjour tout le monde, comment ça va?",
        "expected_answer": (False, "Hello everyone, how are you?")  # French
    },
    {
        "post": "¿Puedes ayudarme con mi tarea de matemáticas?",
        "expected_answer": (False, "Can you help me with my math homework?")  # Spanish
    },
    {
        "post": "Ciao, come stai oggi?",
        "expected_answer": (False, "Hi, how are you today?")  # Italian
    },
    {
        "post": "今日はいい天気ですね",
        "expected_answer": (False, "The weather is nice today, isn't it?")  # Japanese
    },
    {
        "post": "오늘 날씨가 정말 좋네요.",
        "expected_answer": (False, "The weather is really nice today.")  # Korean
    },
    {
        "post": "Привет, как дела? Я изучаю программирование.",
        "expected_answer": (False, "Hi, how are you? I am studying programming.")  # Russian
    },
    {
        "post": "مرحبا، كيف حالك؟ أتمنى أن تكون بخير.",
        "expected_answer": (False, "Hello, how are you? I hope you are doing well.")  # Arabic
    },
    {
        "post": "Olá, tudo bem? Estou aprendendo português.",
        "expected_answer": (False, "Hello, how are you? I am learning Portuguese.")  # Portuguese
    },
    {
        "post": "आज मौसम बहुत अच्छा है।",
        "expected_answer": (False, "The weather is very nice today.")  # Hindi
    },
    {
        "post": "我喜欢学习新语言，这非常有趣。",
        "expected_answer": (False, "I like learning new languages, it is very interesting.")  # Chinese (Simplified)
    },
    {
        "post": "Ik hou van programmeren en nieuwe dingen leren.",
        "expected_answer": (False, "I love programming and learning new things.")  # Dutch
    },
    {
        "post": "Hur mår du idag? Jag hoppas att allt är bra.",
        "expected_answer": (False, "How are you today? I hope everything is well.")  # Swedish
    },
    {
        "post": "Bugün hava çok güzel, dışarı çıkmak istiyorum.",
        "expected_answer": (False, "The weather is very nice today, I want to go outside.")  # Turkish
    },
    {
        "post": "Gdzie jest najbliższa biblioteka?",
        "expected_answer": (False, "Where is the nearest library?")  # Polish
    }
]
abnormal_eval_set = [
    # ── Unintelligible / malformed posts (5) ── bool=False, string = original (untranslatable) ──
    {
        "post": "asdfghjkl qwerty zxcvbnm poiuyt",
        "expected_answer": (False, "asdfghjkl qwerty zxcvbnm poiuyt")  # Random keyboard mash
    },
    {
        "post": "!!@@##$$%%^^&&**((||\\//??>><<",
        "expected_answer": (False, "!!@@##$$%%^^&&**((||\\//??>><<")  # Symbols only
    },
    {
        "post": "Th1s 1s @ t3rr1bly m@lf0rm3d p0st!!!",
        "expected_answer": (False, "Th1s 1s @ t3rr1bly m@lf0rm3d p0st!!!")  # Heavy leet-speak substitution
    },
    {
        "post": "Hello 你好 Bonjour مرحبا Привет 안녕",
        "expected_answer": (False, "Hello 你好 Bonjour مرحبا Привет 안녕")  # Incoherent mixed languages
    },
    {
        "post": "xzqpwv fmrjkl thsdn bnvqwx plzkr",
        "expected_answer": (False, "xzqpwv fmrjkl thsdn bnvqwx plzkr")  # Nonsense non-word tokens
    },
    {
        "post": "aaaaaaaaaaaaaaa bbbbbbbbbbb cccccccc",
        "expected_answer": (False, "aaaaaaaaaaaaaaa bbbbbbbbbbb cccccccc")  # Repeated characters
    },
]

def test_chinese():
    is_english, translated_content = translate_content("这是一条中文消息")
    assert is_english == False
    assert translated_content == "This is a Chinese message"


def eval_single_response_classification(expected_answer: str, llm_response: str) -> float:
  '''TODO: Compares an LLM response to the expected answer from the evaluation dataset using one of the text comparison metrics.'''
  # ----------------- YOUR CODE HERE ------------------ #
  return 1.0 if expected_answer.lower() == llm_response.lower() else 0.0

def eval_single_response_translation(expected_answer: str, llm_response: str) -> float:
  '''TODO: Compares an LLM response to the expected answer from the evaluation dataset using one of the text comparison metrics.'''
  # ----------------- YOUR CODE HERE ------------------ #
  emb1 = model.encode(expected_answer, convert_to_tensor=True)
  emb2 = model.encode(llm_response, convert_to_tensor=True)
  similarity = util.cos_sim(emb1, emb2).item()
  return similarity


def eval_single_response_complete(expected_answer: tuple[bool, str], llm_response: tuple[bool, str]) -> float:
  '''
  Compares an LLM response to the expected answer by combining
  classification accuracy and translation semantic similarity.
  '''
  # 1. Extract components from the tuples
  expected_is_english, expected_text = expected_answer
  llm_is_english, llm_text = llm_response

  # 2. Evaluate the classification (boolean part)
  # We convert to strings to match your existing function signature
  class_score = eval_single_response_classification(
      str(expected_is_english),
      str(llm_is_english)
  )

  # 3. Evaluate the translation/text (string part)
  # This uses your embedding-based cosine similarity model
  trans_score = eval_single_response_translation(
      expected_text,
      llm_text
  )

  # 4. Return an overall score
  # Averaging the two gives equal weight to detection and translation accuracy
  return (class_score + trans_score) / 2.0

def evaluate(query_fn: Callable[[str], str], eval_fn: Callable[[str, str], float], dataset) -> float:
  '''
  TODO: Computes an aggregate score of the chosen evaluation metric across the given dataset. Calls the query_fn function to generate
  LLM outputs for each of the posts in the evaluation dataset, and calls eval_single_response to calculate the metric.
  '''
  # ----------------- YOUR CODE HERE ------------------ #
  total_score = 0

  for item in dataset:
      post = item["post"]
      expected = item["expected_answer"]

      response = query_fn(post)
      score = eval_fn(expected, response)

      total_score += score

  return total_score / len(dataset)

def test_llm_normal_response():
    eval_score = evaluate(translate_content, eval_single_response_complete, normal_eval_set)
    print("Evaluation Score: ", eval_score)
    assert eval_score > 0.3


def test_llm_gibberish_response():
    for i in abnormal_eval_set:
        expected_is_english, expected_text = i["expected_answer"]
        llm_is_english, llm_text = translate_content(i["post"])
        assert expected_is_english == llm_is_english
        assert expected_text == llm_text