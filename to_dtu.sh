rsync --progress -avLK --exclude '__pycache__'  --exclude '*.pth' -e "ssh -A yeza@thinlinc.compute.dtu.dk ssh" . yeza@crius:electra