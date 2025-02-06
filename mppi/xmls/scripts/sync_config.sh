SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
scp -r alexaldermanwebb@129.215.91.20:${1-./brax/brax/scripts/cfg} ${2-$SCRIPT_DIR}
