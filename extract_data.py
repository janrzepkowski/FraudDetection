import zipfile

with zipfile.ZipFile("fraud-transaction-detection.zip", "r") as zip_ref:
    zip_ref.extractall()