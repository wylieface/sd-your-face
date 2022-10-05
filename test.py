import os
import threading
import time
import importlib
from modules import devices
from modules.paths import script_path
import signal
import threading
import base64
from io import BytesIO
from PIL import Image
import time

import modules.codeformer_model as codeformer
import modules.extras
import modules.face_restoration
import modules.gfpgan_model as gfpgan
import modules.img2img

import modules.lowvram
import modules.paths
import modules.scripts
import modules.sd_hijack
import modules.sd_models
import modules.shared as shared
import modules.txt2img
import modules.yftxt2img

import modules.ui
from modules import devices
from modules import modelloader
from modules.paths import script_path
from modules.shared import cmd_opts

modelloader.cleanup_models()
modules.sd_models.setup_model()
codeformer.setup_model(cmd_opts.codeformer_models_path)
gfpgan.setup_model(cmd_opts.gfpgan_models_path)
shared.face_restorers.append(modules.face_restoration.FaceRestoration())
modelloader.load_upscalers()
queue_lock = threading.Lock()
modules.scripts.load_scripts(os.path.join(script_path, "scripts"))

shared.sd_model = modules.sd_models.load_model()

#shared.opts.onchange("sd_model_checkpoint", wrap_queued_call(lambda: modules.sd_models.reload_model_weights(shared.sd_model)))


def run_test(new_prompt):
    print("Running Test")
# def txt2img(prompt: str, negative_prompt: str, prompt_style: str, prompt_style2: str, steps: int, sampler_index: int, restore_faces: bool, tiling: bool, n_iter: int, batch_size: int, cfg_scale: float, seed: int, subseed: int, subseed_strength: float, seed_resize_from_h: int, seed_resize_from_w: int, seed_enable_extras: bool, height: int, width: int, enable_hr: bool, scale_latent: bool, denoising_strength: float, *args):
# {'prompt': 'wyliefox person riding a dinosaur, riding dinosaur, stone age', 
# 'negative_prompt': '', 
# 'prompt_style': 'None', 
# 'prompt_style2': 'None', 
# 'steps': 20, 
# 'sampler_index': 0, 
# 'restore_faces': False, 
# 'tiling': False, 
# 'n_iter': 1, 
# 'batch_size': 1, 
# 'cfg_scale': 7, 
# 'seed': -1.0, 
# 'subseed': -1.0, 
# 'subseed_strength': 0, 
# 'seed_resize_from_h': 0, 
# 'seed_resize_from_w': 0, 
# 'seed_enable_extras': False, 
# 'height': 512, 'width': 512, 
# 'enable_hr': False, 
# 'scale_latent': False, 
# 'denoising_strength': 0.7, 
# 'args': 0, 
# 'p': False, 
# 'processed': False, 
# 'generation_info_js': None}
 
    txt2img_prompt = new_prompt
    txt2img_negative_prompt = ""
    txt2img_prompt_style = ""
    txt2img_prompt_style2 = ""
    steps = 50
    sampler_index = 0
    restore_faces = False
    tiling = False
    batch_count = 1
    batch_size = 1
    cfg_scale = 9
    seed = -1
    subseed = -1
    subseed_strength = 0
    seed_resize_from_h = 0
    seed_resize_from_w = 0
    seed_checkbox = False
    height = 512
    width = 512
    enable_hr = False
    scale_latent = False
    denoising_strength = 0.7
    custom_inputs = None


    inputs=[
        txt2img_prompt,
        txt2img_negative_prompt,
        txt2img_prompt_style,
        txt2img_prompt_style2,
        steps,
        sampler_index,
        restore_faces,
        tiling,
        batch_count,
        batch_size,
        cfg_scale,
        seed,
        subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w, seed_checkbox,
        height,
        width,
        enable_hr,
        scale_latent,
        denoising_strength,
    ]

    print("Inputs: ", inputs)

    output = modules.yftxt2img.yftxt2img(*inputs)

    print("Output: ", output)

    filepath = '/content/' + str(time.time()) + '.jpg'

    output[0][0].save(filepath, 'JPEG')

    #img = Image.fromarray(output[0][0], 'RGB')                  #Crée une image à partir de la matrice
    buffer = BytesIO()
    output[0][0].save(buffer,format="JPEG")                  #Enregistre l'image dans le buffer
    myimage = buffer.getvalue()                     
    #print ("data:image/jpeg;base64,"+base64.b64encode(myimage))
    print ("Out1: ")
    print (myimage)
    print ("Out2: ")
    print (base64.b64encode(myimage))

    # txt2img_args = dict(
    #             fn=wrap_gradio_gpu_call(modules.txt2img.txt2img),
    #             _js="submit",
    #             inputs=[
    #                 txt2img_prompt,
    #                 txt2img_negative_prompt,
    #                 txt2img_prompt_style,
    #                 txt2img_prompt_style2,
    #                 steps,
    #                 sampler_index,
    #                 restore_faces,
    #                 tiling,
    #                 batch_count,
    #                 batch_size,
    #                 cfg_scale,
    #                 seed,
    #                 subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w, seed_checkbox,
    #                 height,
    #                 width,
    #                 enable_hr,
    #                 scale_latent,
    #                 denoising_strength,
    #             ] + custom_inputs,
    #             outputs=[
    #                 txt2img_gallery,
    #                 generation_info,
    #                 html_info
    #             ],
    #             show_progress=False,
    #         )

def run_interactive():
  while 1:
    prompt = input("Prompt: ")
    run_test(prompt)

if __name__ == "__main__":
    run_interactive()
