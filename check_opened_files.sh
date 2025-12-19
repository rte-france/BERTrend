#
# Copyright (c) 2024, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: MPL-2.0
# This file is part of BERTrend.
#
# Script to check which processes open a lot of file descriptors - used to track potential leaks

for pid in /proc/[0-9]*; do
    p=${pid##*/}                                # extract pid number
    fdcount=$(ls "$pid/fd" 2>/dev/null | wc -l)
    cmd=$(tr -d '\0' < "$pid/comm" 2>/dev/null)
    printf "%-6s %-6s %s\n" "$p" "$fdcount" "$cmd"
done | sort -k2 -n

