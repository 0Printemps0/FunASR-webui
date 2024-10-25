# coding=utf-8

import os
import gradio as gr
import numpy as np
import torch
import torchaudio
from funasr import AutoModel
import gc


# MODELSCOPE_CACHE = 

# 加载FunASR模型
def load_model(model_inputs):
	model_abbr = {"热词模型": "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch", "情感模型": "iic/SenseVoiceSmall"}
	model_inputs = "热词模型" if len(model_inputs) < 1 else model_inputs
	model_path = model_abbr[model_inputs]
	model = AutoModel(
		model=model_path,  
		vad_model="fsmn-vad", 
		vad_kwargs={"max_single_segment_time": 30000},
		punc_model="ct-punc", 
		spk_model="cam++",
		device="cuda:0",
		disable_update=True,
	)
	print(f"加载{model_inputs}成功")
	return model

# 模型推理
def model_inference(input_wav, language, fs=16000):
	gc.collect()
	torch.cuda.empty_cache()
	torch.cuda.ipc_collect()
	language_abbr = {"auto": "auto", "zh": "zh", "en": "en", "yue": "yue", "ja": "ja", "ko": "ko", "nospeech": "nospeech"}
	language = "auto" if len(language) < 1 else language
	selected_language = language_abbr[language]
	if isinstance(input_wav, tuple):
		fs, input_wav = input_wav
		input_wav = input_wav.astype(np.float32) / np.iinfo(np.int16).max
		if len(input_wav.shape) > 1:
			input_wav = input_wav.mean(-1)
		if fs != 16000:
			print(f"audio_fs: {fs}")
			resampler = torchaudio.transforms.Resample(fs, 16000)
			input_wav_t = torch.from_numpy(input_wav).to(torch.float32)
			input_wav = resampler(input_wav_t[None, :])[0, :].numpy()
	
	merge_vad = True 
	print(f"language: {language}, merge_vad: {merge_vad}")
	text = model.generate(
		input=input_wav,
		cache={},
		language=selected_language,
		use_itn=True,
		batch_size_s=60, 
		merge_vad=merge_vad,
		merge_length_s=15,
		sentence_timestamp=True,
		hotword='好哥哥',
	)
	
	print(text) #控制台完整输出
	text = text[0]["text"] #页面只输出文字部分
	
	return text



html_content = """
<div>
	<h2 style="font-size: 22px;margin-left: 0px;text-align: center;">FunASR应用程序 FunASR-webui</h2>
</div>
<div style="text-align: center; font-size: 15px; font-weight: bold; color: red; margin-bottom: 20px;">
    ⚠️ 该演示仅供学术研究和体验使用。
</div>
<div style="text-align: center;">
    一键包制作 by 十字鱼|
    <a href="https://space.bilibili.com/893892">🌐 Bilibili</a> 
</div>
"""


def launch():
	global model
	model = load_model("热词模型")
	with gr.Blocks(theme=gr.themes.Soft()) as demo:
		# gr.Markdown(description)
		gr.HTML(html_content)
		with gr.Row():
			with gr.Column():
				audio_inputs = gr.Audio(label="上传音频或使用麦克风")
				
				with gr.Accordion(label="配置"):
					model_inputs = gr.Dropdown(label="模型", choices=["热词模型", "情感模型"], value="热词模型")
					language_inputs = gr.Dropdown(label="语言", choices=["auto", "zh", "en", "yue", "ja", "ko", "nospeech"], value="auto")
				fn_button = gr.Button("开始", variant="primary")
			text_outputs = gr.Textbox(label="结果")
		model_inputs.change(load_model, inputs=model_inputs, outputs=model)
		fn_button.click(model_inference, inputs=[audio_inputs, language_inputs], outputs=text_outputs)

	demo.launch(inbrowser=True, share=False)


if __name__ == "__main__":
	launch()


