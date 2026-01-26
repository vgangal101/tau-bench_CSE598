# Complete Setup Guide: vLLM on ASU Sol for τ-Bench Project

**Course:** CSE 598 - Language Agents for τ-Bench  
**Purpose:** Run Qwen models with vLLM on ASU's Sol supercomputer  
**Time Required:** ~30-45 minutes for first-time setup

---

## Prerequisites

- ✅ ASU student account (ASURITE)
- ✅ Enrolled in CSE 598 course
- ✅ HuggingFace account (free)
- ✅ Basic familiarity with terminal/command line

---

## Part 1: Request ASU HPC Account

### Step 1: Create Your Account

1. Go to: https://links.asu.edu/getHPC
2. Fill out the account request form
3. Use your ASURITE credentials
4. Wait for confirmation email (usually within a few hours)

**Note:** Your account will use your ASURITE ID and password.

---

## Part 2: Install and Connect to VPN

### Step 2: Install Cisco AnyConnect VPN

**For Mac/Windows:**
1. Go to: https://sslvpn.asu.edu
2. Download Cisco AnyConnect for your OS
3. Install the application

**For Linux:**
```bash
# Install OpenConnect (alternative client)
sudo apt install openconnect  # Ubuntu/Debian
# or
sudo yum install openconnect  # RedHat/CentOS
```

### Step 3: Connect to ASU VPN

**Using Cisco AnyConnect (Mac/Windows):**
1. Open Cisco AnyConnect
2. Enter: `sslvpn.asu.edu/2fa`
3. Click "Connect"
4. Enter your ASURITE username
5. Enter your ASURITE password
6. For "second password", type: `push` (not your actual password)
7. Approve the DUO notification on your phone
8. You should see "Connected" ✅

**Using OpenConnect (Linux):**
```bash
sudo openconnect sslvpn.asu.edu/2fa
# Enter ASURITE username
# Enter ASURITE password
# Type: push (for DUO)
# Approve on phone
```

**⚠️ IMPORTANT:** You must be connected to VPN **every time** you want to access Sol.

---

## Part 3: Get HuggingFace Token

### Step 4: Create HuggingFace Token

1. Go to: https://huggingface.co
2. Create an account (or login if you have one)
3. Go to: https://huggingface.co/settings/tokens
4. Click "Create new token"
5. Select **"Read"** token type (NOT Fine-grained)
6. Name it: `cse598` (or anything you like)
7. Click "Create token"
8. **COPY THE TOKEN IMMEDIATELY** (starts with `hf_...`)
9. Save it somewhere safe - you can only see it once!

**Example token format:** `hf_AbCdEfGhIjKlMnOpQrStUvWxYz1234567890`

---

## Part 4: SSH into Sol Supercomputer

### Step 5: Connect to Sol

**Make sure VPN is connected first!**

Open your terminal and type:

```bash
ssh your_asurite@sol.asu.edu
```

Replace `your_asurite` with your actual ASURITE ID (e.g., `jsmith@sol.asu.edu`)

**Enter your ASURITE password when prompted.**

You should see a welcome message and a prompt like:
```
[your_asurite@sol-login02:~]$
```

✅ You're now on Sol!

### Step 6: Quick Orientation

Run these commands to see your environment:

```bash
# See your username
whoami

# See current directory
pwd

# Check home directory space
df -h ~

# List files
ls -la
```

---

## Part 5: Set Up Working Directory

### Step 7: Create Project Directory

```bash
# Create your project workspace in /scratch (more space)
mkdir -p /scratch/$USER/tau-bench-project

# Navigate to it
cd /scratch/$USER/tau-bench-project

# Verify location
pwd
# Should show: /scratch/your_asurite/tau-bench-project
```

**Why /scratch?**
- More space than home directory
- Faster I/O for large model files
- Designed for temporary computational work

---

## Part 6: Set Up Python Environment

### Step 8: Load Mamba and Create Environment

```bash
# Load the mamba module (faster than conda)
module load mamba/latest

# Create environment with Python 3.11
mamba create -n tau-bench python=3.11 -y

# This takes 2-3 minutes - wait for it to complete
```

You'll see progress bars and package installation messages.

### Step 9: Activate Environment

```bash
# Activate the environment
source activate tau-bench

# Your prompt should change to show (tau-bench) at the beginning:
# (tau-bench) [your_asurite@sol-login02:~]$
```

**Verify Python version:**
```bash
python --version
# Should show: Python 3.11.x
```

---

## Part 7: Add HuggingFace Token

### Step 10: Configure HuggingFace Token

```bash
# Open your .bashrc file
nano ~/.bashrc
```

**In nano editor:**
1. Use arrow keys to scroll to the bottom
2. Add this line (replace with YOUR actual token):
   ```bash
   export HF_TOKEN="hf_your_actual_token_here"
   ```
3. Press `Ctrl+X` to exit
4. Press `Y` to save
5. Press `Enter` to confirm

**Reload your bashrc:**
```bash
source ~/.bashrc

# Verify token is set
echo $HF_TOKEN
# Should display your token
```

**⚠️ Keep your token private! Don't share it with anyone.**

---

## Part 8: Install vLLM and Dependencies

### Step 11: Load CUDA Module

```bash
# Load CUDA 12.1.1 (required for vLLM)
module load cuda-12.1.1-gcc-12.1.0

# Verify CUDA is loaded
echo $CUDA_HOME
# Should show a path to CUDA installation
```

### Step 12: Install PyTorch with CUDA Support

```bash
# Install PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

This takes 3-5 minutes. You'll see download progress bars.

### Step 13: Install vLLM

```bash
# Install vLLM
pip install vllm

# Install OpenAI client (for testing)
pip install openai
```

This takes 5-10 minutes. **Be patient!** You'll see compilation messages (lots of them).

### Step 14: Verify Installation

```bash
# Test vLLM import
python -c "import vllm; print('vLLM version:', vllm.__version__)"

# You should see: vLLM version: 0.11.x (or similar)
# Ignore the NVML warning - it's normal on login nodes
```

✅ vLLM is installed!
