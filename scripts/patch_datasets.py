import os
import sys

def patch_datasets_dill():
    try:
        import datasets
    except ImportError:
        print("datasets library is not installed, skipping patch.")
        return

    # Locate _dill.py in datasets/utils/
    datasets_dir = os.path.dirname(datasets.__file__)
    dill_patch_path = os.path.join(datasets_dir, "utils", "_dill.py")

    if not os.path.exists(dill_patch_path):
        print(f"Patch target not found at {dill_patch_path}, skipping patch.")
        return

    with open(dill_patch_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Targets to replace to make it compatible with Python 3.14
    target_def = "def _batch_setitems(self, items):"
    replacement_def = "def _batch_setitems(self, items, *args, **kwargs):"

    target_super = "return super()._batch_setitems(items)"
    replacement_super = "return super()._batch_setitems(items, *args, **kwargs)"

    target_dill = "dill.Pickler._batch_setitems(self, items)"
    replacement_dill = "dill.Pickler._batch_setitems(self, items, *args, **kwargs)"

    modified = False

    if target_def in content:
        content = content.replace(target_def, replacement_def)
        modified = True
    if target_super in content:
        content = content.replace(target_super, replacement_super)
        modified = True
    if target_dill in content:
        content = content.replace(target_dill, replacement_dill)
        modified = True

    if modified:
        try:
            with open(dill_patch_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Successfully applied Python 3.14 serialization patch to: {dill_patch_path}")
        except Exception as e:
            print(f"Failed to write patch to {dill_patch_path}: {e}")
    else:
        # Check if it was already patched
        if replacement_def in content:
            print("Python 3.14 serialization patch is already active.")
        else:
            print("Could not find matching signature pattern in _dill.py to patch.")

if __name__ == "__main__":
    patch_datasets_dill()
