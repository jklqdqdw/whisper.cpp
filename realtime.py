#!/usr/bin/env python3
import queue
import re
import subprocess
import sys
import signal
import threading

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


def transcribe_audio(role, device_id, output_queue):
    print("Start transcribe audio:"+role)
    """
    启动实时语音转录并将结果放入线程安全的队列中。

    参数:
      role: 角色名称 ("interviewer" 或 "candidate")。
      device_id: 音频输入设备ID。
      output_queue: 用于存储输出的线程安全队列。
    """
    # load_translation_model()  # 预加载翻译模型，避免后续重复加载
    cmd = [
        './build/bin/whisper-stream',
        '-m', 'models/ggml-large-v3-turbo.bin',
        '-fa',
        '-kc',
        '--capture', str(device_id),  # 添加音频设备选择参数
    ]
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        last_line = ""
        while True:
            out_line = process.stdout.readline()
            if out_line == '' and process.poll() is not None:
                break
            ansi_escape = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
            out_line = ansi_escape.sub('', out_line)
            out_line = out_line.replace("*Loud music*","").strip()
            if out_line and last_line != out_line:
                last_line = out_line
                if 'kernel_flash_attn_ext_vec_bf16_h256' in out_line or 'init:' in out_line or 'whisper_' in out_line or 'main:' in out_line or '[2K' in out_line or out_line == '.' or out_line == 'Thank you.' or out_line == 'Okay.' or out_line == '[Start speaking]':
                    continue
                output_queue.put({role: out_line})
                print({role: out_line})
    except Exception as ex:
        print(f"{role} 线程发生错误：", ex)
    finally:
        if process.poll() is None:
            process.terminate()

def main():
    output_queue = queue.Queue()  # 创建线程安全的队列

    # 列出音频设备并选择
    list_audio_devices()
    interviewer_device_id = int(input("请选择interviewer音频输入设备ID: "))
    candidate_device_id = int(input("请选择candidate音频输入设备ID: "))

    # 创建并启动子线程
    interviewer_thread = threading.Thread(target=transcribe_audio, args=("interviewer", interviewer_device_id, output_queue))
    candidate_thread = threading.Thread(target=transcribe_audio, args=("candidate", candidate_device_id, output_queue))

    interviewer_thread.start()
    candidate_thread.start()

    # 主线程处理队列中的数据
    try:
        while True:
            if not output_queue.empty():
                output = output_queue.get()
                #print(output)
    except KeyboardInterrupt:
        print("正在终止...")
    finally:
        interviewer_thread.join()
        candidate_thread.join()

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
    main()