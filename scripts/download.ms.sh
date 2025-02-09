download_from="https://www.modelscope.cn"
model_dir="$HOME/data"
modelid_list=(
    Qwen/Qwen2.5-0.5B
    Qwen/Qwen2.5-0.5B-Instruct
    Qwen/Qwen2.5-1.5B
    Qwen/Qwen2.5-1.5B-Instruct
    Qwen/Qwen2.5-7B
    Qwen/Qwen2.5-7B-Instruct
    deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
    deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
)


if [ ! -d $model_dir ]; then
    mkdir -p $model_dir
fi

cd $model_dir
for mi in "${modelid_list[@]}"; do
    git lfs install
    git clone $download_from/${mi}.git
done