#
# Copyright (c) 2024, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: MPL-2.0
# This file is part of BERTrend.
#

CUDA_VISIBLE_DEVICES=0 OPENAI_API_KEY=$AZURE_WATTELSE_OPENAI_API_KEY OPENAI_ENDPOINT=$AZURE_WATTELSE_OPENAI_ENDPOINT OPENAI_DEFAULT_MODEL_NAME=$AZURE_WATTELSE_OPENAI_DEFAULT_MODEL_NAME streamlit run --theme.primaryColor royalblue summarization/summarizer_app.py