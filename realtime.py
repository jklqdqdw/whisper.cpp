#!/usr/bin/env python3
import re
import subprocess
import sys
import signal

translation_model = None
translation_tokenizer = None

def load_translation_model():
    global translation_model, translation_tokenizer
    if translation_model is None or translation_tokenizer is None:
        from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
        translation_model = MBartForConditionalGeneration.from_pretrained("SnypzZz/Llama2-13b-Language-translate")
        translation_tokenizer = MBart50TokenizerFast.from_pretrained("SnypzZz/Llama2-13b-Language-translate", src_lang="en_XX")

def translate_text(text, target_lang):
    """
    翻译给定英文文本到目标语言。

    参数:
      text: 输入的英文文本.
      target_lang: 目标语言代码，例如 "hi_IN" 表示印地语，"zh_CN" 表示中文.
    返回:
      翻译后的文本字符串.
    """
    global translation_model, translation_tokenizer
    if translation_model is None or translation_tokenizer is None:
        load_translation_model()
    model_inputs = translation_tokenizer(text, return_tensors="pt")
    generated_tokens = translation_model.generate(
        **model_inputs,
        forced_bos_token_id=translation_tokenizer.lang_code_to_id[target_lang]
    )
    translated_texts = translation_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return translated_texts[0] if translated_texts else ""


def main(cpatureID):
    load_translation_model()  # 预加载翻译模型，避免后续重复加载
    # 构造 whisper-stream 的命令行参数
    # 根据需要可以调整参数，比如模型路径、线程数、step 和 length
    cmd = [
        './build/bin/whisper-stream',
        '-m', 'models/ggml-large-v3-turbo.bin',
        '-fa',
        '-kc',
        '--capture', str(cpatureID),  # 添加音频设备选择参数

    ]

    print("启动实时语音识别，请开始说话，按 Ctrl+C 退出...")
    try:
        # 启动子进程，并实时读取标准输出
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)

        # 注册 SIGINT (Ctrl+C) 的处理函数
        def signal_handler(sig, frame):
            print("正在终止...")
            process.terminate()
            sys.exit(0)
        signal.signal(signal.SIGINT, signal_handler)

        # 实时输出子进程的输出
        last_line = ""
        while True:
            out_line = process.stdout.readline()
            if out_line == '' and process.poll() is not None:
                break
            ansi_escape = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
            out_line = ansi_escape.sub('', out_line)
            # 再来 strip()
            out_line = out_line.strip()
            if out_line:
                if last_line == out_line:  # 直接比较去除空白后的字符串
                    continue
                else:
                    last_line = out_line  # 更新 last_line
                if 'kernel_flash_attn_ext_vec_bf16_h256' in out_line or 'init:' in out_line or 'whisper_' in out_line or 'main:' in out_line or '[2K' in out_line or out_line == '.' or out_line == 'Thank you.' or out_line == 'Okay.' or out_line == '[Start speaking]':
                    continue
                print("line:" + out_line)  # 翻译输出文本
                translated_text = translate_text(out_line, "zh_CN")
                print("翻译: ", translated_text)
    except Exception as ex:
        print("发生错误：", ex)
    finally:
        if process.poll() is None:
            process.terminate()

def list_audio_devices():
    import sounddevice as sd
    """
    列出系统所有的音频设备
    """
    devices = sd.query_devices()
    print("\n=== 音频输入设备 ===")
    index=0
    for device in devices:
        if device['max_input_channels'] > 0:
            print(f"[{index}] {device['name']}")
            print(f"    通道数: {device['max_input_channels']}")
            print(f"    采样率: {device['default_samplerate']}Hz")
            index+=1


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='实时语音识别和翻译程序')
    list_audio_devices()
    # 让用户选择音频输入设备
    device_id = int(input("请选择音频输入设备ID: "))
    main(device_id)