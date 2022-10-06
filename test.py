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

def run_test(new_prompt):

    print("Running Test")
    txt2img_prompt = new_prompt
    txt2img_negative_prompt = ""
    txt2img_prompt_style = ""
    txt2img_prompt_style2 = ""
    steps = 50
    sampler_index = 0
    restore_faces = True
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
    buffer = BytesIO()
    output[0][0].save(buffer,format="JPEG")
    myimage = buffer.getvalue()                     
    print ("Out1: ")
    print (myimage)
    print ("Out2: ")
    print (base64.b64encode(myimage))



def run_interactive():
  while 1:
    prompt = input("Prompt: ")
    run_test(prompt)

if __name__ == "__main__":
    run_interactive()
