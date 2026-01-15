# Training Data Generation - Setup & Execution Guide

**Status**: Production-scale configurations ready for GCP execution  
**Last Updated**: December 30, 2025

## Overview

StrandWeaver requires large-scale training data for 5 AI subsystems:
1. **EdgeWarden** - Overlap edge classifier (Random Forests)
2. **PathWeaver** - Graph neural network path predictor
3. **DiploidDetangler** - Haplotype classifier
4. **ThreadCompass** - Ultra-long read router
5. **SVScribe** - Structural variant detector

This guide covers training data generation at production scale (100 genomes × 10 Mb per scenario).

---

## 🎯 Production Configurations

All production scenarios now use:
- **Genome size**: 10 Mb (realistic for training)
- **Num genomes**: 100 (sufficient for model convergence)
- **Total per scenario**: 1 Gb genome data

### Available Scenarios

| Scenario | Focus | Special Features |
|----------|-------|------------------|
| `balanced` | Standard training | Balanced coverage, standard complexity |
| `repeat_heavy` | Repeat resolution | 60% repeats, k=51 |
| `sv_dense` | SV detection | 10× SV density, max 500kb SVs |
| `diploid_focus` | Haplotype phasing | 2% heterozygosity, 40× Hi-C |
| `ultra_long_focus` | UL routing | 50× UL coverage |

---

## 💻 Hardware Requirements

### Local Testing (MacBook)
```
- RAM: 16+ GB
- CPU: 8+ cores
- Time: 2-5 minutes for simple test
- Suitable for: Testing workflow only
```

### Production (GCP VM - Recommended)
```
Machine: n1-highmem-16 + NVIDIA T4 GPU
- vCPUs: 16 cores
- RAM: 104 GB
- GPU: T4 (16 GB VRAM)
- Storage: 500 GB SSD
- Cost: ~$1.50/hour
- Time per scenario: 8-15 hours
- Total for all 5: ~50-60 hours ($75-90)
```

---

## 🚀 Quick Start

### Step 1: Test Locally (5 minutes)

```bash
# Test that workflow is working
python test_training_workflow.py --test

# Show production configurations
python test_training_workflow.py --show-production

# Show all scenarios
python test_training_workflow.py --show-scenarios
```

If the test passes, you're ready for GCP!

---

### Step 2: Setup GCP VM

```bash
# Set project
gcloud config set project YOUR_PROJECT_ID

# Create VM
gcloud compute instances create strandweaver-training \
    --machine-type=n1-highmem-16 \
    --zone=us-central1-a \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --maintenance-policy=TERMINATE \
    --boot-disk-size=500GB \
    --boot-disk-type=pd-ssd \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --metadata="install-nvidia-driver=True" \
    --scopes=cloud-platform

# SSH into VM
gcloud compute ssh strandweaver-training --zone=us-central1-a
```

---

### Step 3: Install Dependencies (On VM)

```bash
# System packages
sudo apt-get update
sudo apt-get install -y python3.11 python3-pip git nvidia-driver-535

# Python packages
pip3 install --upgrade pip
pip3 install numpy scipy pandas scikit-learn \
    torch torchvision torchaudio \
    pytorch-geometric pyg-lib torch-scatter torch-sparse \
    cupy-cuda12x biopython pysam

# Clone repository
cd ~
git clone https://github.com/YOUR_USERNAME/strandweaver.git
cd strandweaver
pip3 install -e .

# Verify GPU
nvidia-smi  # Should show Tesla T4
```

---

### Step 4: Generate Training Data (On VM)

#### Option A: All Scenarios at Once

```bash
cd ~/strandweaver

# Dry run (show estimates)
python gcp_generate_all_training_data.py --all --dry-run

# Run all 5 scenarios (50-60 hours)
python gcp_generate_all_training_data.py --all --workers 12
```

#### Option B: One Scenario at a Time

```bash
# Run specific scenario
python gcp_generate_all_training_data.py \
    --scenario balanced \
    --workers 12 \
    --output training_data

# Check progress (separate terminal)
watch -n 60 'du -sh training_data/*'
```

---

### Step 5: Monitor Progress

```bash
# In separate SSH session
cd ~/strandweaver

# Watch output size
watch -n 30 'du -sh training_data/*/; echo ""; df -h | grep /dev/sda1'

# Watch GPU usage
watch -n 5 nvidia-smi

# Check logs
tail -f nohup.out  # If using nohup
```

---

### Step 6: Download Results

```bash
# On VM: Compress data
cd ~
tar -czf training_data_$(date +%Y%m%d).tar.gz training_data/

# Check size
ls -lh training_data_*.tar.gz

# On local machine: Download
gcloud compute scp \
    strandweaver-training:~/training_data_*.tar.gz \
    ~/Downloads/ \
    --zone=us-central1-a

# Alternative: Use Cloud Storage
# On VM:
gsutil -m cp -r ~/training_data gs://YOUR_BUCKET/strandweaver/

# On local:
gsutil -m cp -r gs://YOUR_BUCKET/strandweaver/training_data ~/Downloads/
```

---

### Step 7: Stop VM

```bash
# Stop (keeps disk, can restart)
gcloud compute instances stop strandweaver-training --zone=us-central1-a

# Delete completely (careful!)
gcloud compute instances delete strandweaver-training --zone=us-central1-a
```

---

## 📊 Expected Outputs

### Per Scenario

```
training_data/
└── balanced/
    ├── per_genome/
    │   ├── genome_0000/
    │   │   ├── features.pkl       # Extracted features
    │   │   ├── labels.pkl         # Ground truth labels
    │   │   └── metadata.json      # Genome metadata
    │   ├── genome_0001/
    │   └── ...
    ├── train/
    │   ├── shard_0000.jsonl
    │   ├── shard_0001.jsonl
    │   └── ...
    ├── val/
    │   └── shard_0000.jsonl
    ├── test/
    │   └── shard_0000.jsonl
    └── metadata.json              # Dataset metadata
```

### Dataset Statistics

Per scenario (100 genomes × 10 Mb):
- **Genome files**: ~2 GB (FASTA)
- **Read files**: ~10 GB (FASTQ)
- **Features**: ~5-8 GB (PKL)
- **Total**: ~15-20 GB per scenario

All 5 scenarios: **~75-100 GB total**

---

## 🔧 Troubleshooting

### Out of Memory

```bash
# Reduce workers
python gcp_generate_all_training_data.py --scenario balanced --workers 8

# Or reduce genome size temporarily
# Edit main_training_workflow.py, set genome_size=5_000_000
```

### GPU Not Detected

```bash
# Check driver
nvidia-smi

# Reinstall
sudo apt-get install -y nvidia-driver-535
sudo reboot
```

### Slow Performance

```bash
# Check I/O
iostat -x 5

# Check CPU
htop

# Check GPU
nvidia-smi -l 5
```

### Disk Full

```bash
# Check space
df -h

# Clean up
rm -rf training_data/*/per_genome/*/reads/  # Remove intermediate reads
```

---

## 💰 Cost Optimization

### Preemptible VMs (70% discount)

```bash
# Add to create command
--preemptible \
--no-restart-on-failure

# Cost: ~$0.45/hour vs $1.50/hour
# Risk: VM can be shut down anytime
# Best for: Non-critical runs
```

### Spot VMs (Similar to preemptible)

```bash
--provisioning-model=SPOT
```

### Multi-VM Parallel

Run scenarios on separate VMs in parallel:

```bash
# VM 1: balanced
# VM 2: repeat_heavy
# VM 3: sv_dense
# VM 4: diploid_focus
# VM 5: ultra_long_focus

# Total time: 15 hours (max of all) vs 60 hours (sequential)
# Total cost: Similar ($75-90) but much faster
```

---

## 📈 Next Steps After Data Generation

1. **Verify datasets**
   ```bash
   python -c "from strandweaver.training import load_jsonl_shard; \
              data = load_jsonl_shard('training_data/balanced/train/shard_0000.jsonl'); \
              print(f'Loaded {len(data)} examples')"
   ```

2. **Begin Phase 5.4: ML Model Training**
   - EdgeWarden: XGBoost/Random Forest
   - PathWeaver: PyTorch Geometric GNN
   - DiploidDetangler: PyTorch MLP
   - ThreadCompass: PyTorch LSTM
   - SVScribe: Ensemble (XGBoost + CNN)

3. **Model training scripts** (coming in Phase 5.4):
   ```bash
   python scripts/train_edge_classifier.py --data training_data/balanced
   python scripts/train_path_gnn.py --data training_data/balanced
   python scripts/train_diploid_model.py --data training_data/diploid_focus
   python scripts/train_ul_router.py --data training_data/ultra_long_focus
   python scripts/train_sv_detector.py --data training_data/sv_dense
   ```

---

## 📞 Support

- **Issues**: Open GitHub issue
- **Questions**: Check `docs/PHASE5_3_SUMMARY.md`
- **Logs**: Save VM output with `tee`:
  ```bash
  python gcp_generate_all_training_data.py --all --workers 12 2>&1 | tee generation.log
  ```

---

## ✅ Checklist

Before running on GCP:
- [ ] Local test passes (`python test_training_workflow.py --test`)
- [ ] GCP project configured
- [ ] Billing enabled
- [ ] Sufficient quota (16 vCPUs + 1 GPU)
- [ ] Repository synced to latest

During execution:
- [ ] GPU detected (`nvidia-smi`)
- [ ] Training starts successfully
- [ ] Monitor disk space (`df -h`)
- [ ] Monitor progress (`du -sh training_data/*`)

After completion:
- [ ] All scenarios completed
- [ ] Data downloaded
- [ ] VM stopped
- [ ] Datasets verified
- [ ] Ready for Phase 5.4 (model training)
