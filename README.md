# brilliant_automation

### Getting Started

Clone this repository:

```bash
git clone https://github.com/Brilliant-Automation/capstone.git
cd capstone
```

### Create the Environment
Create the environment from the environment.yml file:

```bash
conda env create -f environment.yml
conda activate brilliant-auto-env
```

### Run Preprocessing Script

The preprocessing script processes raw sensor and ratings data and outputs a merged dataset for further analysis.

#### **Supported Devices**
The script currently supports the following devices:
1. `1#High-Temp Fan`
2. `8#Belt Conveyer`
3. `Tube Mill`

#### **Usage**
Run the script from the root directory:

```
python model/src/preprocess.py --device "<device_name>" [--data_dir <data_directory>] [--output_dir <output_directory>]
```

##### **Arguments**:
- `--device` (Required): Specify one of the supported device names (e.g. `8#Belt Conveyer`).
- `--data_dir` (Optional): Directory containing raw `.xlsx` files. Defaults to `Data/raw`.
- `--output_dir` (Optional): Directory to save the processed CSV file. Defaults to `Data/process`.

#### **Examples**
1. To process data for `8#Belt Conveyer` using default directories:
   ```bash
   python model/src/preprocess.py --device "8#Belt Conveyer"
   ```
   Output:
   ```
   Data/process/8#Belt Conveyer_merged.csv
   ```

2. To process data for `Tube Mill` using custom directories:
   ```bash
   python model/src/preprocess.py --device "Tube Mill" --output_dir custom_data/processed
   ```


### Run the EDA Notebook
The exploratory data analysis for the conveyor belt data is done using a Jupyter notebook:

```
jupyter notebook notebook/eda_conveyer_belt.ipynb
```

This makes sure all plots are generated dynamically by running all cells in order. 

### Generate PDF

To generate the proposal report as a PDF, ensure you have [Quarto](https://quarto.org/) installed. Then run the following command in your terminal from the root directory:

```
quarto render docs/proposal.qmd --to pdf
```

This will generate `proposal.pdf` in the `docs` directory.
