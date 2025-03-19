SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

find . -name policy_params_* -exec stat -c '%W %n' {} + | sort | head -n 1 | awk '{print $2}'
