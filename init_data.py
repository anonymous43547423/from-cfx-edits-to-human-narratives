import gdown

url = "https://drive.google.com/drive/folders/1YguaEn6U4N3u4t0K_3hSNTDRsCkSIqpr"

tmp_folder = "."

gdown.download_folder(url, output=tmp_folder, quiet=False, use_cookies=False)  # type: ignore[attr-defined]
