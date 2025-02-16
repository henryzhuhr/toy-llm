# Get the base image
FROM ubuntu:24.04

# ============================================================
#   安装基础的环境 (分为开发环境和部署环境)
# 「基本系统」软件列表 APT_INSTALL_LIST
#                   - ubuntu-desktop-minimal    桌面环境（会比较大）
#                   - ubuntu-server             服务器环境 (包含 openssh-server ?)
#                   - ca-certificates   证书
#                   - locales           语言环境
#                   - sudo              用于提权
# 「开发环境」软件列表 APT_INSTALL_DEV_LIST
#                   - vim nano          编辑器 (ubuntu-server 包含 vi)
#                   - tree              目录树
#                   - pciutils          PCI 设备信息
#                   - git               版本控制工具
#                   - openssh-client    SSH 客户端 (ubuntu-server 不包含，所以如果使用 git clone 需要安装)
#                   - python3-dev python3-pip python3.12-venv   Python 环境
# ============================================================
# 是否开发模式，如果是开发模式，会安装「开发环境」中的软件
ARG DEV_MODE=1

# 「基本系统」中需要的软件列表
ARG APT_INSTALL_LIST="ubuntu-server"
ARG APT_INSTALL_LIST="$APT_INSTALL_LIST ca-certificates locales sudo"

# 「开发环境」中需要的软件列表
ARG APT_INSTALL_DEV_LIST=""
ARG APT_INSTALL_DEV_LIST="$APT_INSTALL_DEV_LIST vim nano tree pciutils"
ARG APT_INSTALL_DEV_LIST="$APT_INSTALL_DEV_LIST git openssh-client"
ARG APT_INSTALL_DEV_LIST="$APT_INSTALL_DEV_LIST build-essential cmake"
ARG APT_INSTALL_DEV_LIST="$APT_INSTALL_DEV_LIST python3-pip python3-venv"


# 自定义镜像源 (tuna 在自定义网络下不可访问)
# ARG MIRRORS_URL="mirrors.tuna.tsinghua.edu.cn"
ARG MIRRORS_URL="mirrors.ustc.edu.cn"
ENV DEBIAN_FRONTEND=noninteractive

# set -x 选项会使得每个命令在执行前都会被打印出来，报错时会显示是哪个命令出错
RUN set -x && \
    sed -i 's/#force_color_prompt=yes/force_color_prompt=yes/g' ${HOME}/.bashrc && \
    sed -i -e "s#//.*archive.ubuntu.com#//${MIRRORS_URL}#g" -e "s#//ports.ubuntu.com#//${MIRRORS_URL}#g" /etc/apt/sources.list.d/ubuntu.sources && \
    apt-get update && \
    apt-get upgrade -y && \
    # 开发模式下「基本系统」和「部署环境」的软件列表合并
    if [ $DEV_MODE -eq 1 ]; then { APT_INSTALL_LIST="$APT_INSTALL_LIST $APT_INSTALL_DEV_LIST"; } fi && \
    apt-get install -y --no-install-recommends ${APT_INSTALL_LIST} && \
    apt-get autoremove -y && \
    # 非开发模式下清理 apt 缓存
    if [ $DEV_MODE -ne 1 ]; then { apt-get clean all; rm -rf /var/lib/apt/lists/*; } fi && \
    localedef -i zh_CN -c -f UTF-8 -A /usr/share/locale/locale.alias zh_CN.UTF-8 

ENV LANG=zh_CN.utf8

# 系统时间错误可能导致: `tls: failed to verify certificate: x509: certificate signed by unknown authority`
# 修改时区 (date -R 检查当前时间)
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

# 启用 bash 补全
RUN sed -i.bak '/^#if ! shopt -oq posix; then$/,/^#fi$/s/^#//' /etc/bash.bashrc

ARG USER_NAME="root"
ARG USER_HOME="/${USER_NAME}"

# ============================================================
#   Python 环境
# ============================================================
# # PEP 668 – Python base environments Python 增强提案 (PEP)：https://realpython.com/python-virtual-environments-a-primer/?ref=yaolong.net
# # https://www.adamsdesk.com/posts/resolve-pip-externally-managed-environment
# # 临时解决方案  --ignoreExternallyManagedEnvironment
ADD . $USER_HOME/tmp
ENV PYTHON_VENV_DIR=$USER_HOME/.venv
ENV PIP_MIRROR="https://mirrors.ustc.edu.cn/pypi/simple"
RUN set -x && \
    python3 -m venv ${PYTHON_VENV_DIR} --clear && \
    echo "source ${PYTHON_VENV_DIR}/bin/activate" >> $USER_HOME/.bashrc && \
    VENV_PYTHON=${PYTHON_VENV_DIR}/bin/python3 && \
    $VENV_PYTHON -m pip install -i $PIP_MIRROR -U --upgrade pip setuptools wheel && \
    $VENV_PYTHON -m pip config set global.index-url $PIP_MIRROR && \
    [ -f $USER_HOME/tmp/requirements.txt ] && $VENV_PYTHON -m pip install -r ${USER_HOME}/tmp/requirements.txt || echo "requirements.txt not found, skip install" && \
    sudo rm -rf $USER_HOME/tmp && \
    # 清理缓存，缓存目录： - [Linux]: ~/.cache/pip and ~/.cache/pipenv  - [macOS]: ~/Library/Caches/pip and ~/.cache/pipenv  - [Windows]: %LocalAppData%/pip/Cache
    $VENV_PYTHON -m pip cache purge

# 提升虚拟环境 python 的优先级
ENV PATH=${PYTHON_VENV_DIR}/bin:$PATH
ENV VIRTUAL_ENV_DISABLE_PROMPT=1