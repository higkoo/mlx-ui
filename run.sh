#!/bin/bash
. ~/Applications/conda.env # 按你的实际需要修改
conda activate mlx # 按你的实际情况修改

# 默认可配置参数
PORT=2233

# 获取本地实际 IP 地址
function get_local_ip() {
    # 尝试不同的方法获取本地 IP 地址
    local ip
    # 方法 1: 使用 ifconfig
    ip=$(ifconfig | grep -E 'inet (addr:)?([0-9]+\.){3}[0-9]+' | grep -v '127.0.0.1' | head -n 1 | awk '{print $2}' | sed 's/addr://')
    if [[ -n "$ip" ]]; then
        echo "$ip"
        return
    fi
    # 方法 2: 使用 ip 命令
    ip=$(ip addr | grep -E 'inet (?!127\.0\.0\.1)' | head -n 1 | awk '{print $2}' | cut -d'/' -f1)
    if [[ -n "$ip" ]]; then
        echo "$ip"
        return
    fi
    # 方法 3: 使用 hostname 命令
    ip=$(hostname -I | awk '{print $1}')
    if [[ -n "$ip" ]]; then
        echo "$ip"
        return
    fi
    # 如果所有方法都失败，使用 localhost
    echo "localhost"
}

# 初始化地址为本地实际 IP
ADDRESS=$(get_local_ip)

# 初始化其他参数数组
OTHER_ARGS=()

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            if [[ -n "$2" ]]; then
                PORT="$2"
                shift # 移动到下一个参数
            fi
            shift
            ;;
        --address)
            if [[ -n "$2" ]]; then
                ADDRESS="$2"
                shift # 移动到下一个参数
            fi
            shift
            ;;
        *)
            # 保留其他参数
            OTHER_ARGS+=("$1")
            shift
            ;;
    esac
done

set -x
# 启动 Streamlit 应用
streamlit run app.py --server.port="$PORT" --server.address="$ADDRESS" -- "${OTHER_ARGS[@]}"
