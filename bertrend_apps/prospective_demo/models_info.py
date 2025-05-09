#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import multiprocessing
import os
import random
import shutil
import sys
import time

import pandas as pd
import streamlit as st
import toml
from loguru import logger

from bertrend import BERTREND_LOG_PATH, BEST_CUDA_DEVICE, load_toml_config
from bertrend.demos.demos_utils.icons import (
    EDIT_ICON,
    DELETE_ICON,
    WARNING_ICON,
    INFO_ICON,
    TOGGLE_ON_ICON,
    TOGGLE_OFF_ICON,
    ERROR_ICON,
    RESTART_ICON,
)
from bertrend_apps.common.crontab_utils import (
    check_cron_job,
    remove_from_crontab,
    add_job_to_crontab,
)
from bertrend_apps.prospective_demo import (
    get_user_models_path,
    get_model_cfg_path,
    DEFAULT_ANALYSIS_CFG,
)
from bertrend_apps.prospective_demo.perf_utils import get_least_used_gpu
from bertrend_apps.prospective_demo.process_new_data import regenerate_models
from bertrend_apps.prospective_demo.streamlit_utils import clickable_df


@st.fragment
def models_monitoring():
    if not st.session_state.user_feeds:
        st.stop()

    st.session_state.model_analysis_cfg = {}
    displayed_list = []

    for model_id in sorted(st.session_state.user_feeds.keys()):
        try:
            st.session_state.model_analysis_cfg[model_id] = load_toml_config(
                get_model_cfg_path(
                    user_name=st.session_state.username, model_id=model_id
                )
            )
        except Exception:
            st.session_state.model_analysis_cfg[model_id] = DEFAULT_ANALYSIS_CFG
        list_models = get_models_info(model_id)
        displayed_list.append(
            {
                "id": model_id,
                "# modèles": len(list_models) if list_models else 0,
                "date 1er modèle": list_models[0] if list_models else None,
                "date dernier modèle": list_models[-1] if list_models else None,
                "fréquence mise à jour (# jours)": st.session_state.model_analysis_cfg[
                    model_id
                ]["model_config"]["granularity"],
                "fenêtre d'analyse (# jours)": st.session_state.model_analysis_cfg[
                    model_id
                ]["model_config"]["window_size"],
            }
        )

    st.session_state.models_paths = {
        model_id: get_user_models_path(st.session_state.username, model_id)
        for model_id in st.session_state.user_feeds.keys()
    }

    df = pd.DataFrame(displayed_list)
    if not df.empty:
        df = df.sort_values(by="id", inplace=False).reset_index(drop=True)

    clickable_df_buttons = [
        (EDIT_ICON, edit_model_parameters, "secondary"),
        (lambda x: toggle_icon(df, x), toggle_learning, "secondary"),
        (DELETE_ICON, handle_delete_models, "primary"),
        (RESTART_ICON, handle_regenerate_models, "primary"),
    ]
    clickable_df(df, clickable_df_buttons)


@st.dialog("Paramètres")
def edit_model_parameters(row_dict: dict):
    model_id = row_dict["id"]
    st.write(f"**Paramètres des modèles pour la veille {model_id}**")

    new_granularity = st.slider(
        "Fréquence de mise à jour des modèles (en jours)",
        min_value=1,
        max_value=30,
        value=st.session_state.model_analysis_cfg[model_id]["model_config"][
            "granularity"
        ],
        step=1,
        help=f"{INFO_ICON} Sélection de la fréquence à laquelle la détection de sujets est effectuée. "
        f"Le nombre de jours sélectionné doit être choisi pour s'assurer d'un volume de données suffisant.",
    )
    new_window_size = st.slider(
        "Sélection de la fenêtre temporelle (en jours)",
        min_value=new_granularity,
        max_value=30,
        value=max(
            st.session_state.model_analysis_cfg[model_id]["model_config"][
                "window_size"
            ],
            new_granularity,
        ),
        step=1,
        help=f"{INFO_ICON} Sélection de la plage temporelle considérée pour calculer les différents "
        f"types de signaux (faibles, forts)",
    )

    st.write(f"**Paramètres d'analyse de la veille {model_id}: éléments à inclure**")
    topic_evolution = st.checkbox(
        "Evolution du sujet",
        value=st.session_state.model_analysis_cfg[model_id]["analysis_config"][
            "topic_evolution"
        ],
    )
    evolution_scenarios = st.checkbox(
        "Scénarios d'évolution",
        value=st.session_state.model_analysis_cfg[model_id]["analysis_config"][
            "evolution_scenarios"
        ],
    )
    multifactorial_analysis = st.checkbox(
        "Analyse multifactorielle",
        value=st.session_state.model_analysis_cfg[model_id]["analysis_config"][
            "multifactorial_analysis"
        ],
    )

    model_config = {
        "granularity": new_granularity,
        "window_size": new_window_size,
        "language": (
            "French"
            if st.session_state.user_feeds[model_id]["data-feed"]["language"] == "fr"
            else "English"
        ),
    }
    analysis_config = {
        "topic_evolution": topic_evolution,
        "evolution_scenarios": evolution_scenarios,
        "multifactorial_analysis": multifactorial_analysis,
    }

    if st.button("OK"):
        save_model_config(
            model_id, {"model_config": model_config, "analysis_config": analysis_config}
        )
        st.rerun()


def save_model_config(model_id: str, config: dict):
    model_cfg_path = get_model_cfg_path(
        user_name=st.session_state.username, model_id=model_id
    )
    with open(model_cfg_path, "w") as toml_file:
        toml.dump(config, toml_file)
    logger.debug(f"Saved model analysis config {config} to {model_cfg_path}")


def load_model_config(model_id: str) -> dict:
    model_cfg_path = get_model_cfg_path(
        user_name=st.session_state.username, model_id=model_id
    )
    try:
        return load_toml_config(model_cfg_path)
    except Exception:
        return DEFAULT_ANALYSIS_CFG


@st.dialog("Confirmation")
def handle_delete_models(row_dict: dict):
    """Function to handle remove models from cache"""
    model_id = row_dict["id"]
    st.warning(
        f":orange[{WARNING_ICON}] Voulez-vous vraiment supprimer tous les modèles stockés pour la veille **{model_id}** ?"
    )
    col1, col2, _ = st.columns([2, 2, 8])
    with col1:
        if st.button("Oui", type="primary"):
            remove_scheduled_training_for_user(
                model_id=model_id, user=st.session_state.username
            )
            delete_cached_models(model_id)
            logger.info(f"Modèles en cache supprimés pour la veille {model_id} !")
            time.sleep(0.2)
            st.rerun()
    with col2:
        if st.button("Non"):
            st.rerun()


@st.dialog("Regénération des modèles")
def handle_regenerate_models(row_dict: dict):
    """Function to regenerate models from scratch"""
    model_id = row_dict["id"]
    st.warning(
        f"{WARNING_ICON} Voulez-vous re-générer l'ensemble des modèles pour la veille {model_id} ?"
    )
    st.warning(
        f"{WARNING_ICON} L'ensemble des modèles existant pour cette veille sera supprimé."
    )
    st.error(
        f"{ERROR_ICON} Attention, cette regénération ne peut pas être annulée une fois lancée !"
    )
    col1, col2, _ = st.columns([2, 2, 8])
    with col1:
        if yes_btn := st.button("Oui", type="primary"):
            # Delete previously stored model
            delete_cached_models(model_id)
            logger.info(f"Modèles en cache supprimés pour la veille {model_id} !")

            # Regenerate new models
            # Launch model generation in a separate thread to avoid blocking the app
            # Create a new process with parameters
            # Set the start method to 'spawn' - required for a process using CUDA
            os.environ["CUDA_VISIBLE_DEVICES"] = get_least_used_gpu()
            multiprocessing.set_start_method("spawn", force=True)

            process = multiprocessing.Process(
                target=regenerate_models,
                kwargs={"model_id": model_id, "user": st.session_state.username},
            )
            # Start the process
            process.start()

            time.sleep(0.2)
            # st.rerun()

    with col2:
        if st.button("Non"):
            st.rerun()

    if yes_btn:
        st.info(
            f"{INFO_ICON} Regénération en cours des modèles pour la veille {model_id}. "
            "L'opération peut prendre un peu de temps."
        )
        st.info(f"{INFO_ICON} Vous pouvez fermer cette fenêtre.")


def toggle_learning(cfg: dict):
    """Activate / deactivate the learning from the crontab"""
    model_id = cfg["id"]
    if check_if_learning_active_for_user(
        model_id=model_id, user=st.session_state.username
    ):
        if remove_scheduled_training_for_user(
            model_id=model_id, user=st.session_state.username
        ):
            st.toast(
                f"Le learning pour la veille **{model_id}** est déactivé !",
                icon=INFO_ICON,
            )
            logger.info(f"Learning pour {model_id} désactivé !")
    else:
        schedule_training_for_user(model_id, st.session_state.username)
        st.toast(
            f"Le learning pour la veille **{model_id}** est activé !", icon=WARNING_ICON
        )
        logger.info(f"Learning pour {model_id} activé !")
    time.sleep(0.2)
    st.rerun()


def toggle_icon(df: pd.DataFrame, index: int) -> str:
    """Switch the toggle icon depending on the statis of the scrapping feed in the crontab"""
    model_id = df["id"][index]
    return (
        f":green[{TOGGLE_ON_ICON}]"
        if check_if_learning_active_for_user(
            model_id=model_id, user=st.session_state.username
        )
        else f":red[{TOGGLE_OFF_ICON}]"
    )


def check_if_learning_active_for_user(model_id: str, user: str):
    """Checks if a given scrapping feed is active (registered in the crontab"""
    if user:
        return check_cron_job(rf"process_new_data train-new-model.*{user}.*{model_id}")
    else:
        return False


def remove_scheduled_training_for_user(model_id: str, user: str):
    """Removes from the crontab the training job matching the provided model_id"""
    if user:
        return remove_from_crontab(
            rf"process_new_data train-new-model.*{user}.*{model_id}"
        )


def schedule_training_for_user(model_id: str, user: str):
    """Schedule data scrapping on the basis of a feed configuration file"""
    schedule = generate_crontab_expression(
        st.session_state.model_analysis_cfg[model_id]["model_config"]["granularity"]
    )
    logpath = BERTREND_LOG_PATH / "users" / user
    logpath.mkdir(parents=True, exist_ok=True)
    command = (
        f"{sys.prefix}/bin/python -m bertrend_apps.prospective_demo.process_new_data train-new-model {user} {model_id} "
        f"> {logpath}/learning_{model_id}.log 2>&1"
    )
    env_vars = f"CUDA_VISIBLE_DEVICES={BEST_CUDA_DEVICE}"
    add_job_to_crontab(schedule, command, env_vars)


def delete_cached_models(model_id: str):
    """Removes models from the cache"""
    # Remove the directory and all its contents
    shutil.rmtree(st.session_state.models_paths[model_id])


def generate_crontab_expression(days_interval: int) -> str:
    # Random hour between 0 and 6 (inclusive)
    hour = random.randint(0, 6)  # run during the night
    # Random minute rounded to the nearest 10
    minute = random.choice([0, 10, 20, 30, 40, 50])
    # Compute days
    days = [str(i) for i in range(1, 31, days_interval)]
    # Crontab expression format: minute hour day_of_month month day_of_week
    crontab_expression = f"{minute} {hour} {','.join(days)} * *"
    return crontab_expression


def safe_timestamp(x: str) -> pd.Timestamp | None:
    try:
        return pd.Timestamp(x)
    except Exception as e:
        return None


def get_models_info(model_id: str) -> list:
    """Returns the list of topic models that are stored, identified by their timestamp"""
    user_model_dir = get_user_models_path(st.session_state.username, model_id)
    if not user_model_dir.exists():
        return []
    matching_files = user_model_dir.glob(r"????-??-??")
    return sorted(
        [
            safe_timestamp(x.name)
            for x in matching_files
            if safe_timestamp(x.name) is not None
        ]
    )
