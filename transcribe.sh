#!/bin/bash

# 检查参数
if [ "$#" -lt 1 ]; then
    echo "用法: $0 <input_file> [model_name]"
    echo "示例: $0 input/audio.mp3 base.en"
    exit 1
fi

# 设置变量
INPUT_FILE="$1"
MODEL="ggml-large-v3-turbo.bin" # 默认使用base模型
INPUT_DIR="input"
OUTPUT_DIR="output"
MODELS_DIR="models"
TEMP_DIR="temp"

# 创建必要的目录
mkdir -p "$OUTPUT_DIR" "$TEMP_DIR"

# 检查输入文件是否存在
if [ ! -f "$INPUT_FILE" ]; then
    echo "错误: 输入文件 '$INPUT_FILE' 不存在"
    exit 1
fi

# 获取文件名(不含扩展名)
FILENAME=$(basename "$INPUT_FILE")
FILENAME_NO_EXT="${FILENAME%.*}"

# 转换音频为whisper可接受的格式
# - 采样率: 16000 Hz
# - 声道: 单声道
# - 格式: 16-bit PCM WAV
echo "正在转换音频格式..."
TEMP_WAV="$TEMP_DIR/${FILENAME_NO_EXT}.wav"
ffmpeg -i "$INPUT_FILE" -ar 16000 -ac 1 -c:a pcm_s16le "$TEMP_WAV" -y

# 检查模型文件是否存在,不存在则下载
MODEL_PATH="$MODELS_DIR/$MODEL"
if [ ! -f "$MODEL_PATH" ]; then
    echo "正在下载模型 $MODEL..."
    sh ./models/download-ggml-model.sh "$MODEL"
fi

# 运行whisper转录
echo "正在转录音频..."
OUTPUT_FILE="$OUTPUT_DIR/${FILENAME_NO_EXT}.txt"
./build/bin/whisper-cli -m "$MODEL_PATH" -f "$TEMP_WAV" -otxt > "$OUTPUT_FILE"

# 检查转录是否成功
if [ $? -eq 0 ]; then
    echo "转录完成! 输出文件: $OUTPUT_FILE"
    # 清理临时文件
    rm -f "$TEMP_WAV"
else
    echo "转录失败!"
    exit 1
fi