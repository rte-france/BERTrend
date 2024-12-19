#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

# Success Messages
STATE_SAVED_MESSAGE = "Application state saved."
STATE_RESTORED_MESSAGE = "Application state restored."
MODELS_SAVED_MESSAGE = "Models saved."
MODELS_RESTORED_MESSAGE = "Models restored."
EMBEDDINGS_CALCULATED_MESSAGE = "Embeddings calculated successfully!"
MODEL_TRAINING_COMPLETE_MESSAGE = "Model training complete!"
MODEL_MERGING_COMPLETE_MESSAGE = "Model merging complete!"
TOPIC_COUNTS_SAVED_MESSAGE = "Topic and signal counts saved to {file_path}"
CACHE_PURGED_MESSAGE = "Cache purged."

PROGRESS_BAR_DESCRIPTION = "Batches processed"

# Error Messages
NO_DATA_WARNING = "No data available for the selected granularity."
NO_MODELS_WARNING = "No saved models found."
NO_CACHE_WARNING = "No cache found."
TOPIC_NOT_FOUND_WARNING = (
    "Topic {topic_number} not found in the merge histories within the specified window."
)
NO_GRANULARITY_WARNING = "Granularity value not found."
NO_DATASET_WARNING = "Please select at least one dataset to proceed."
