# this scripts installs necessary requirements and launches main program in webui.py
import subprocess
import os
import sys
import importlib.util
import shlex

## second part of script
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
import gdown

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

print("Running warm-build.py")

dir_repos = "repositories"
dir_tmp = "tmp"

python = sys.executable
git = os.environ.get('GIT', "git")
torch_command = os.environ.get('TORCH_COMMAND', "pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113")
# requirements_file = os.environ.get('REQS_FILE', "requirements_versions.txt")
requirements_file = "requirements.txt"
commandline_args = os.environ.get('COMMANDLINE_ARGS', "")

gfpgan_package = os.environ.get('GFPGAN_PACKAGE', "git+https://github.com/TencentARC/GFPGAN.git@8d2447a2d918f8eba5a4a01463fd48e45126a379")
clip_package = os.environ.get('CLIP_PACKAGE', "git+https://github.com/openai/CLIP.git@d50d76daa670286dd6cacf3bcd80b5e4823fc8e1")

stable_diffusion_commit_hash = os.environ.get('STABLE_DIFFUSION_COMMIT_HASH', "69ae4b35e0a0f6ee1af8bb9a5d0016ccb27e36dc")
taming_transformers_commit_hash = os.environ.get('TAMING_TRANSFORMERS_COMMIT_HASH', "24268930bf1dce879235a7fddd0b2355b84d7ea6")
k_diffusion_commit_hash = os.environ.get('K_DIFFUSION_COMMIT_HASH', "a7ec1974d4ccb394c2dca275f42cd97490618924")
codeformer_commit_hash = os.environ.get('CODEFORMER_COMMIT_HASH', "c5b4593074ba6214284d6acd5f1719b6c5d739af")
blip_commit_hash = os.environ.get('BLIP_COMMIT_HASH', "48211a1594f1321b00f14c9f7a5b4813144b2fb9")

args = shlex.split(commandline_args)


def extract_arg(args, name):
    return [x for x in args if x != name], name in args


args, skip_torch_cuda_test = extract_arg(args, '--skip-torch-cuda-test')


def repo_dir(name):
    return os.path.join(dir_repos, name)


def run(command, desc=None, errdesc=None):
    if desc is not None:
        print(desc)

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

    if result.returncode != 0:

        message = f"""{errdesc or 'Error running command'}.
Command: {command}
Error code: {result.returncode}
stdout: {result.stdout.decode(encoding="utf8", errors="ignore") if len(result.stdout)>0 else '<empty>'}
stderr: {result.stderr.decode(encoding="utf8", errors="ignore") if len(result.stderr)>0 else '<empty>'}
"""
        raise RuntimeError(message)

    return result.stdout.decode(encoding="utf8", errors="ignore")


def run_python(code, desc=None, errdesc=None):
    return run(f'"{python}" -c "{code}"', desc, errdesc)


def run_pip(args, desc=None):
    return run(f'"{python}" -m pip {args} --prefer-binary', desc=f"Installing {desc}", errdesc=f"Couldn't install {desc}")


def check_run(command):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    return result.returncode == 0


def check_run_python(code):
    return check_run(f'"{python}" -c "{code}"')


def is_installed(package):
    try:
        spec = importlib.util.find_spec(package)
    except ModuleNotFoundError:
        return False

    return spec is not None


def git_clone(url, dir, name, commithash=None):
    # TODO clone into temporary dir and move if successful

    if os.path.exists(dir):
        return

    run(f'"{git}" clone "{url}" "{dir}"', f"Cloning {name} into {dir}...", f"Couldn't clone {name}")

    if commithash is not None:
        run(f'"{git}" -C {dir} checkout {commithash}', None, "Couldn't checkout {name}'s hash: {commithash}")


try:
    commit = run(f"{git} rev-parse HEAD").strip()
except Exception:
    commit = "<none>"

print(f"Python {sys.version}")
print(f"Commit hash: {commit}")


if not is_installed("torch") or not is_installed("torchvision"):
    run(f'"{python}" -m {torch_command}', "Installing torch and torchvision", "Couldn't install torch")

if not skip_torch_cuda_test:
    run_python("import torch; assert torch.cuda.is_available(), 'Torch is not able to use GPU; add --skip-torch-cuda-test to COMMANDLINE_ARGS variable to disable this check'")

if not is_installed("gfpgan"):
    run_pip(f"install {gfpgan_package}", "gfpgan")

if not is_installed("clip"):
    run_pip(f"install {clip_package}", "clip")

os.makedirs(dir_repos, exist_ok=True)

git_clone("https://github.com/CompVis/stable-diffusion.git", repo_dir('stable-diffusion'), "Stable Diffusion", stable_diffusion_commit_hash)
git_clone("https://github.com/CompVis/taming-transformers.git", repo_dir('taming-transformers'), "Taming Transformers", taming_transformers_commit_hash)
git_clone("https://github.com/crowsonkb/k-diffusion.git", repo_dir('k-diffusion'), "K-diffusion", k_diffusion_commit_hash)
git_clone("https://github.com/sczhou/CodeFormer.git", repo_dir('CodeFormer'), "CodeFormer", codeformer_commit_hash)
git_clone("https://github.com/salesforce/BLIP.git", repo_dir('BLIP'), "BLIP", blip_commit_hash)

if not is_installed("lpips"):
    run_pip(f"install -r {os.path.join(repo_dir('CodeFormer'), 'requirements.txt')}", "requirements for CodeFormer")

run_pip(f"install -r {requirements_file}", "requirements for Web UI")

sys.argv += args

print("End of first stage")

print("Downloading warming model")
id = "1mSY7Z_8_PZxa5s7Ge8vaQTnHEdrGazsO"
test_model_filepath = "models/Stable-diffusion/wyliefox_model.ckpt"
gdown.download(id=id, output=test_model_filepath, quiet=False)

print("Downloading facial reconstruction model")
id = "1Xi4uN4ro3CP49BPl6O4vqQ-Jz4va-GAS"
facial_model_filepath = "GFPGANv1.4.pth"
gdown.download(id=id, output=facial_model_filepath, quiet=False)

print("Setting up models")
modelloader.cleanup_models()
modules.sd_models.setup_model()
codeformer.setup_model(cmd_opts.codeformer_models_path)
gfpgan.setup_model(cmd_opts.gfpgan_models_path)
shared.face_restorers.append(modules.face_restoration.FaceRestoration())
modelloader.load_upscalers()
queue_lock = threading.Lock()
modules.scripts.load_scripts(os.path.join(script_path, "scripts"))

print("Loading model")
shared.sd_model = modules.sd_models.load_model()

txt2img_prompt = "(wyliefox person) dressed as a king giving thumbs up"
txt2img_negative_prompt = ""
txt2img_prompt_style = ""
txt2img_prompt_style2 = ""
steps = 20
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

print("Running test prompt")
output = modules.yftxt2img.yftxt2img(*inputs)

# print("Saving test image")
# filepath = 'test-image.jpg'
# output[0][0].save(filepath, 'JPEG')

print("Test generation complete")

if "--no-delete" not in args:
    print("Removing warmup model")
    os.remove(test_model_filepath)

print("Warm build complete")
exit(0)