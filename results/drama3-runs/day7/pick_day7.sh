#!/bin/bash
# Day 7: Whisper small + base scoring of the narration retries (retry2 takes) + the canonical seed-0 takes
/opt/ep0/run.sh "PYTHONPATH=/opt/ep0/pyext-whisper:\$PYTHONPATH python -m models.demos.blackhole.drama3.tests.day7_retry_pick /opt/ep0/day6/narration_retry/variants.json /opt/ep0/day6/narration /opt/ep0/day6/narration_retry /opt/ep0/day6/narration_retry/picks_day7.json"
echo "EXIT $?"
