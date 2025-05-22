#!/usr/bin/env bash
set -o errexit

# Install system dependencies
apt-get update
apt-get install -y ffmpeg libavcodec-extra

# Install Python dependencies  
pip install -r requirements.txt