def get_prompt_for_mistral(classes, with_rag=False):
    prompt_start = "<s>[INST] Ты профессиональный разметчик данных. Тебе нужно размечать тексты которые я присылаю. Я буду присылать Текст, а ты будешь присылать мне подходящую Метку. "
    prompt_label_info = ""
    for class_name, class_label in classes.items():
        prompt_label_part_info = f", напиши метку '{class_label}', если этот текст относится к классу '{class_name}'"
        prompt_label_info += prompt_label_part_info
    prompt_label_info += '. '
    prompt_label_info = "Н" + prompt_label_info[3:]
    prompt_rag_part = "Ты можешь использовать следующие примеры размеченных текстов. Изучи их схожесть и используй их как подсказку, если тексты похожи, и только если они помогут тебе правильно определить метку: {context}. "
    prompt_label_model_answer = "[/INST]Хорошо. Я"
    for class_name, class_label in classes.items():
        prompt_label_model_answer_part = f" буду присылать Метку '{class_label}' если посчитаю что Текст относится к классу '{class_name}',"
        prompt_label_model_answer += prompt_label_model_answer_part
    prompt_label_model_answer = prompt_label_model_answer[:-1] + '.'
    prompt_end = '</s> [INST] Текст:"""Привет. Как дела?""". Метка: [/INST]0</s> [INST] Текст: """{question}""". Метка: [/INST]'
    if with_rag:
        prompt = prompt_start + prompt_label_info + prompt_rag_part + prompt_label_model_answer + prompt_end
        return prompt
    else:
        prompt = prompt_start + prompt_label_info + prompt_label_model_answer + prompt_end
        return prompt