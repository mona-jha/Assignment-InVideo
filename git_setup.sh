#!/bin/bash

# ...existing code...

echo -e "\n===== Adding All Files and Folders to Git ====="
# This ensures all files and folders, including hidden ones, are tracked
git add -A
echo "All files and folders have been staged for commit"

# Commit with a meaningful message
echo -e "\n===== Committing All Changes ====="
git commit -m "Add all project files and folders"
echo "All changes have been committed"

# ...existing code...

echo -e "\n===== Pushing All Content to GitHub ====="
read -p "Push all files and folders to GitHub now? (y/n): " confirm
if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
    git push -u origin main
    echo "All project files and folders have been successfully pushed to GitHub"
else
    echo "Push canceled. Push all content later with: git push -u origin main"
fi

# ...existing code...
