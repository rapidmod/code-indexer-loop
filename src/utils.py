import hashlib


# changed "hash_md5" to "md5" to avoid shadowing the built-in "hashlib.md5"
# rapidmod.io - 04/04/2024
def hash_md5(filename):
    md5 = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    return md5.hexdigest()
