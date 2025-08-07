#!/bin/bash

echo "🔧 Setting up your fork for deployment..."

# Get your GitHub username
echo "Enter your GitHub username:"
read GITHUB_USERNAME

echo "Adding your fork as remote..."
git remote add fork https://github.com/$GITHUB_USERNAME/bajaj_finserv.git

echo "Pushing vedica-2 branch to your fork..."
git push fork vedica-2

echo "✅ Done! Now you can deploy from your fork in Railway:"
echo "Repository: $GITHUB_USERNAME/bajaj_finserv"
echo "Branch: vedica-2"
