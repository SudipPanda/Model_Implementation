from pathlib import Path
import sys

# Add parent directory
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))


from model.mistral.modeling_mistral import load_mistral_config


if __name__ == "__main__":
    config = load_mistral_config()
    print(config)