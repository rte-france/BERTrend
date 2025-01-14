#!/bin/bash

#
# Copyright (c) 2024, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: MPL-2.0
# This file is part of BERTrend.
#

# Set logs directory and create if not exists
export BERTREND_LOGS_DIR=$BERTREND_BASE_DIR/logs/bertrend
mkdir -p $BERTREND_LOGS_DIR

echo "Starting Wattelse Veille & Analyse"
screen -dmS curebot bash -c 'cd `pwd`/exploration/curebot && ./start_newsletter_generator.sh 2>&1 | tee -a $BERTREND_LOGS_DIR/curebot.log; bash'
