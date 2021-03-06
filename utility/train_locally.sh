#!/bin/bash
set -e

AICROWD_DATA_ENABLED="YES"
# replace the following with the python executable path in your conda env
PYTHON_EXECUTABLE="python3"

if [[ " $@ " =~ " --no-data " ]]; then
   AICROWD_DATA_ENABLED="NO"
else
    $PYTHON_EXECUTABLE ./utility/verify_or_download_data.py
fi



EXTRAOUTPUT=" > /dev/null 2>&1 "
if [[ " $@ " =~ " --verbose " ]]; then
   EXTRAOUTPUT=""
fi

export PYRO_SERIALIZERS_ACCEPTED='pickle'
export PYRO_SERIALIZER='pickle'

# Run local name server
eval "pyro4-ns $EXTRAOUTPUT &"
trap "kill -11 $! > /dev/null 2>&1;" EXIT

# Run instance manager to generate performance report
export EVALUATION_STAGE='manager'
eval "$PYTHON_EXECUTABLE run.py --seeds 1 $EXTRAOUTPUT &"
trap "kill -11 $! > /dev/null 2>&1;" EXIT

# Run the training phase
sleep 2
echo "RUNNING TRAINING!"
export MINERL_INSTANCE_MANAGER_REMOTE="1"
export EVALUATION_STAGE='training'
export EVALUATION_RUNNING_ON='local'
export EXITED_SIGNAL_PATH='shared/training_exited'
rm -f $EXITED_SIGNAL_PATH
export ENABLE_AICROWD_JSON_OUTPUT='False'
eval "$PYTHON_EXECUTABLE run.py $EXTRAOUTPUT && touch $EXITED_SIGNAL_PATH || touch $EXITED_SIGNAL_PATH &"
trap "kill -11 $! > /dev/null 2>&1;" EXIT

# View the evaluation state
export ENABLE_AICROWD_JSON_OUTPUT='True'
$PYTHON_EXECUTABLE utility/parser.py || true
kill $(jobs -p)
