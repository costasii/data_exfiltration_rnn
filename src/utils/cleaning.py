from pathlib import Path

# Folder containing the .pth files
folder = Path("/home/mirta/dev/data_exfiltration/results/centralised/hdfs")

for file in folder.rglob("deeplog_epoch*.pth"):
    if file.is_file():
        print(f"Deleting: {file}")
        file.unlink()

print("Done.")