#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import re

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from bertrend.config.parameters import EMBEDDING_DEVICE
from bertrend.services.summarizer import Summarizer


# DEFAULT_ABSTRACTIVE_MODEL = "mrm8488/camembert2camembert_shared-finetuned-french-summarization"
# DEFAULT_ABSTRACTIVE_MODEL = "csebuetnlp/mT5_multilingual_XLSum"
# NB. Les deux modèles précédents ont des tendances à halluciner en rajoutant des informations qui ne sont pas dans le texte de départ !
DEFAULT_ABSTRACTIVE_MODEL = "facebook/mbart-large-50"


class AbstractiveSummarizer(Summarizer):
    ## class that performs auto summary using T multi langual model
    def __init__(self, model_name=DEFAULT_ABSTRACTIVE_MODEL):
        self.model_name = model_name
        self.WHITESPACE_HANDLER = lambda k: re.sub(
            "\s+", " ", re.sub("\n+", " ", k.strip())
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model = self.model.to(EMBEDDING_DEVICE)

    def generate_summary(self, article_text, **kwargs) -> str:
        return self.summarize_batch([article_text])[0]

    def summarize_batch(self, article_texts: list[str], **kwargs) -> list[str]:
        inputs = self.tokenizer(
            [self.WHITESPACE_HANDLER(text) for text in article_texts],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        )

        input_ids = inputs.input_ids.to(EMBEDDING_DEVICE)

        attention_mask = inputs.attention_mask.to(EMBEDDING_DEVICE)

        max_length = 512

        output_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            no_repeat_ngram_size=2,
            num_beams=4,
        )

        summaries = [
            self.tokenizer.decode(
                output_id, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            for output_id in output_ids
        ]
        return summaries


if __name__ == "__main__":
    text = "Apollo 11 est une mission du programme spatial américain Apollo au cours de laquelle, pour la première fois, des hommes se sont posés sur la Lune, le lundi 21 juillet 1969. L'agence spatiale américaine, la NASA, remplit ainsi l'objectif fixé par le président John F. Kennedy en 1961 de poser un équipage sur la Lune avant la fin de la décennie 1960. Il s'agissait de démontrer la supériorité des États-Unis sur l'Union soviétique qui avait été mise à mal par les succès soviétiques au début de l'ère spatiale dans le contexte de la guerre froide qui oppose alors ces deux pays. Ce défi est lancé alors que la NASA n'a pas encore placé en orbite un seul astronaute. Grâce à une mobilisation de moyens humains et financiers considérables, l'agence spatiale rattrape puis dépasse le programme spatial soviétique."

    model_name = "mrm8488/camembert2camembert_shared-finetuned-french-summarization"

    summarizer = AbstractiveSummarizer(model_name)
    summary = summarizer.generate_summary(text)

    print(summary)
