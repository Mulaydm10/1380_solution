#!/bin/bash

# Check if dvc is installed
if ! command -v dvc &> /dev/null; then
    echo "Please install dvc using pip3 install dvc[s3]"
    exit 1
fi

# Create .dvc directory if it doesn't exist
mkdir -p .dvc

# Check if dvc_key.txt exists, if not create it with random 32-character string
if [ ! -f ".dvc/dvc_key.txt" ]; then
    # Generate random 32-character string using Python
    random_key=$(python3 -c "import secrets; import string; print(''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(32)))")
    echo "$random_key" > .dvc/dvc_key.txt
    echo "Created new dvc_key.txt with random key"
else
    echo "dvc_key.txt already exists, skipping creation"
fi

# Modify dvc remote with the key from dvc_key.txt
dvc remote modify yandex-storage url "s3://ml-cup-dvc/$(cat .dvc/dvc_key.txt)"

echo "Success."
echo "To commit run following steps:"
echo "dvc add big_model.pth"
echo "dvc push"
echo "git add ."
echo "git commit -m 'My wonderful solution'"
echo "git push"