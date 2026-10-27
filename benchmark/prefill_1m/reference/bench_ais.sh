LEN=${1:-${LEN}}
PORT=30000

export no_proxy=localhost,127.0.0.1
# same as the serve script: bench + profile HTTP is local-only, keep the proxy out
unset HTTP_PROXY http_proxy HTTPS_PROXY https_proxy ALL_PROXY all_proxy

. profile_dir_name_${PORT}.sh

echo ${SGLANG_TORCH_PROFILER_DIR}
DATE_NAME=$(date +"%F-%H%M%S")
PREFIX=${DATE_NAME}
OUTPUT_PATH_LAST=${PREFIX}_aisbench_${LEN}
OUTPUT_PATH=${SGLANG_TORCH_PROFILER_DIR}/${OUTPUT_PATH_LAST}
mkdir ${OUTPUT_PATH}
cp $0 ${OUTPUT_PATH}
LOG_NAME=${OUTPUT_PATH}/${PREFIX}_aisbench.log
echo LOG_NAME=${LOG_NAME}

echo "bash $0 $@" 2>&1 | tee -a ${LOG_NAME}
npu-smi info 2>&1 | tee -a ${LOG_NAME}
env 2>&1 | tee -a ${LOG_NAME}
pip list -v 2>&1 | tee -a ${LOG_NAME}

PARAMS="--input_len ${LEN} --output_len 1 --request_rate 0 --dataset_type prefix_cache --prefix_test --seed 4"
PARAMS="${PARAMS} --repeat_rate 0.99"
#PARAMS="${PARAMS} --repeat_rate 0.9"
PARAMS="${PARAMS} --data_num 16 --concurrency 16"
#PARAMS="${PARAMS} --data_num 4 --concurrency 1"

echo "python3 aisbench_test.py ${PARAMS}" 2>&1 | tee -a ${LOG_NAME}
cd /home/krodyushkin/aisbench_auto_tools_prefix
python3 aisbench_test.py ${PARAMS} 2>&1 | tee -a ${LOG_NAME}




#curl -X POST 127.0.0.1:30000/start_profile -H "Content-Type: application/json" -d '{"output_dir": "'${OUTPUT_PATH}'"}'

#python3 aisbench_test.py ${PARAMS} 2>&1 | tee -a ${LOG_NAME}

#curl -X POST 127.0.0.1:30000/stop_profile -H "Content-Type: application/json" -d '{"output_dir": "'${OUTPUT_PATH}'"}'
#curl -X POST 127.0.0.1:30000/stop_profile -H "Content-Type: application/json"
