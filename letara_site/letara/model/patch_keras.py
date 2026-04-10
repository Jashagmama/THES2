import json
import shutil
import zipfile
from pathlib import Path

MODEL_FILES = ["handwriting_MNIST.keras", "hwv1.keras"]

def patch_obj(obj):
    if isinstance(obj, dict):
        class_name = obj.get("class_name")
        config = obj.get("config")

        if isinstance(config, dict):
            # Fix InputLayer fields
            if class_name == "InputLayer":
                if "batch_shape" in config:
                    batch_shape = config.pop("batch_shape")
                    config["batch_input_shape"] = batch_shape

                # Remove unsupported fields
                config.pop("optional", None)

            # Remove unsupported quantization config
            config.pop("quantization_config", None)

            # Convert dtype object to plain string
            dtype_val = config.get("dtype")
            if isinstance(dtype_val, dict):
                dtype_config = dtype_val.get("config", {})
                dtype_name = dtype_config.get("name")
                if dtype_name:
                    config["dtype"] = dtype_name

        for k, v in list(obj.items()):
            obj[k] = patch_obj(v)
        return obj

    if isinstance(obj, list):
        return [patch_obj(x) for x in obj]

    return obj

def patch_keras_file(model_path_str):
    model_path = Path(model_path_str)
    if not model_path.exists():
        print(f"SKIP: {model_path} not found")
        return

    temp_dir = model_path.with_suffix(".tmpdir")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()

    with zipfile.ZipFile(model_path, "r") as zf:
        zf.extractall(temp_dir)

    config_path = temp_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json in {model_path}")

    data = json.loads(config_path.read_text(encoding="utf-8"))
    patched = patch_obj(data)
    config_path.write_text(json.dumps(patched), encoding="utf-8")

    patched_path = model_path.with_name(model_path.stem + "_patched.keras")
    if patched_path.exists():
        patched_path.unlink()

    with zipfile.ZipFile(patched_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in temp_dir.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(temp_dir))

    shutil.rmtree(temp_dir)
    print(f"Created: {patched_path}")

for mf in MODEL_FILES:
    patch_keras_file(mf)
