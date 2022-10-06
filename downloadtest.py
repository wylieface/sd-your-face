# In this file, we define download_model
# It runs during container build time to get model weights built into the container


# from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
import os
import git

def download_sd():
    print("Downloading stable diffusion from github")
    working_dir = os.getcwd()
    print("Working dir: " + working_dir)
    git.Repo.clone_from('https://github.com/wylieface/sd-your-face', 'sd')
    print("Repo cloned")

    


    # # do a dry run of loading the huggingface model, which will download weights at build time
    # #Set auth token which is required to download stable diffusion model weights
    # HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")

    # lms = LMSDiscreteScheduler(
    #     beta_start=0.00085, 
    #     beta_end=0.012, 
    #     beta_schedule="scaled_linear"
    # )

    # model = StableDiffusionPipeline.from_pretrained(
    #     "CompVis/stable-diffusion-v1-4", 
    #     scheduler=lms,
    #     use_auth_token=HF_AUTH_TOKEN
    # )

if __name__ == "__main__":
    download_sd()
    os.chdir('sd')
    import warmbuild