import json

def get_messages_texts_from_json(json_path):
    texts = []

    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)

        for message in data['messages']:
            message_text = message['text']

            text = ''

            if type(message_text) == list:
                for text_raw in message_text:
                    if type(text_raw) == dict:
                        text += ' ' + text_raw['text']
                    else:
                        text += text_raw
            else:
                text = message_text

            texts.append(text)

    return texts