import subprocess
from pathlib import Path

# List of Hugging Face dataset or model repos to clone
REPOS = [
    "https://huggingface.co/datasets/Amod/mental_health_counseling_conversations",
    #"https://huggingface.co/datasets/yuntian-deng/Dialogue-Emotion-Dataset",  # Add more here
]

# Directory where repos will be cloned
DEST_DIR = Path("hf_git_clones")
DEST_DIR.mkdir(exist_ok=True)

def git_clone(url):
    repo_name = url.rstrip("/").split("/")[-1]
    target_path = DEST_DIR / repo_name

    if target_path.exists():
        print(f"🔁 Skipping {repo_name} (already cloned)")
        return

    try:
        print(f"📥 Cloning {repo_name}...")
        subprocess.run(["git", "clone", url, str(target_path)], check=True)
        print(f"✅ Cloned {repo_name} into {target_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to clone {url}: {e}")

if __name__ == "__main__":
    for repo_url in REPOS:
        git_clone(repo_url)