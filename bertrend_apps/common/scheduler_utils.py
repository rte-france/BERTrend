#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import locale
import random
from abc import abstractmethod, ABC
from pathlib import Path

from cron_descriptor import (
    Options,
    CasingTypeEnum,
    ExpressionDescriptor,
    DescriptionTypeEnum,
)

from bertrend import BEST_CUDA_DEVICE
from bertrend.demos.demos_utils.i18n import get_current_internationalization_language


class SchedulerUtils(ABC):

    @staticmethod
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

    @staticmethod
    def get_understandable_cron_description(cron_expression: str) -> str:
        """Returns a human understandable crontab description."""
        # Save current locale
        saved_locale = locale.setlocale(locale.LC_ALL)

        options = Options()
        options.casing_type = CasingTypeEnum.Sentence
        options.use_24hour_time_format = True

        locale_code = (
            "fr_FR.UTF-8"
            if get_current_internationalization_language() == "fr"
            else "en_US.UTF-8"
        )
        crontab_locale_code = (
            "fr_FR" if get_current_internationalization_language() == "fr" else "en"
        )

        options.locale_code = crontab_locale_code

        try:
            # Set temporary locale to specific locale
            locale.setlocale(locale.LC_ALL, locale_code)
            descriptor = ExpressionDescriptor(cron_expression, options)
            description = descriptor.get_description(DescriptionTypeEnum.FULL)

        finally:
            # Restore original locale
            locale.setlocale(locale.LC_ALL, saved_locale)

        return description

    @abstractmethod
    def add_job_to_crontab(
        self, schedule: str, command: str, env_vars=None, command_kwargs: dict = None
    ) -> bool:
        """Add the specified job to the crontab."""
        pass

    @abstractmethod
    def schedule_scrapping(self, feed_cfg: Path, user: str = None):
        """Schedule data scrapping on the basis of a feed configuration file"""
        pass

    @abstractmethod
    def schedule_newsletter(
        self,
        newsletter_cfg_path: Path,
        data_feed_cfg_path: Path,
        cuda_devices: str = BEST_CUDA_DEVICE,
    ):
        """Schedule data scrapping on the basis of a feed configuration file"""
        pass

    @abstractmethod
    def schedule_training_for_user(self, schedule: str, model_id: str, user: str):
        """Schedule data scrapping on the basis of a feed configuration file"""
        pass

    @abstractmethod
    def schedule_report_generation_for_user(
        self, schedule: str, model_id: str, user: str, report_config: dict
    ) -> bool:
        """Schedule automated report generation based on model configuration"""
        pass

    @abstractmethod
    def remove_scrapping_for_user(self, feed_id: str, user: str | None = None):
        """Removes from the scheduler service the job matching the provided feed_id"""
        pass

    @abstractmethod
    def remove_scheduled_training_for_user(self, model_id: str, user: str):
        """Removes from the crontab the training job matching the provided model_id"""
        pass

    @abstractmethod
    def remove_scheduled_report_generation_for_user(
        self, model_id: str, user: str
    ) -> bool:
        """Removes from the crontab the report generation job matching the provided model_id"""
        pass

    @abstractmethod
    def check_if_scrapping_active_for_user(
        self, feed_id: str, user: str | None = None
    ) -> bool:
        """Checks if a given scrapping feed is active (registered with the service)."""
        pass

    @abstractmethod
    def check_if_learning_active_for_user(self, model_id: str, user: str):
        """Checks if a given scrapping feed is active (registered in the crontab"""
        pass

    @abstractmethod
    def check_if_report_generation_active_for_user(
        self, model_id: str, user: str
    ) -> bool:
        """Checks if automated report generation is active (registered in the crontab)"""
        pass

    def update_scheduled_training_for_user(self, model_id: str, user: str):
        """Updates the crontab with the new training job"""
        if self.check_if_learning_active_for_user(model_id, user):
            self.remove_scheduled_training_for_user(model_id, user)
            self.schedule_training_for_user(model_id, user)
            return True
        return False

    def update_scheduled_report_generation_for_user(
        self, model_id: str, user: str
    ) -> bool:
        """Updates the crontab with the new report generation job"""
        if self.check_if_report_generation_active_for_user(model_id, user):
            self.remove_scheduled_report_generation_for_user(model_id, user)
            return self.schedule_report_generation_for_user(model_id, user)
        return False
