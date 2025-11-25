## Module Structure
```
dicty_config.py       # Configuration and argument parsing
dicty_utils.py        # Data loading and processing utilities
dicty_dataset.py      # PyTorch dataset classes
dicty_models.py       # Neural network model definitions
dicty_train.py        # Training functions
dicty_evaluate.py     # Evaluation functions
dicty_main.py         # Main pipeline orchestrator
```

### Step 1: Verify Configuration
dicty_config.py

Check that paths are correct

### Step 2: Test Data Loading
dicty_utils.py

Verify data loads correctly

### Step 3: Test Dataset Creation
python dicty_dataset.py

Verify batching works

### Step 4: Test Model Architecture
python dicty_models.py

Verify models run without errors

### Step 5: Test Training Loop
dicty_train.py

Verify training and saving works

### Step 6: Test Evaluation
dicty_evaluate.py

Verify metrics computation

### Step 7: Run Full Pipeline
python dicty_main.py

Verify everything works together

### Step 8: Full Training

dicty_main.py