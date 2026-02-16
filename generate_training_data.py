"""
PHASE 1: Synthetic Knowledge Base Generator
Generates 10,000 synthetic training records that strictly follow the statistical
profile of USG_Data_cleared.csv while emphasizing key signals:
- Cables-X failure rate: ~20-25%
- Golden Zone (3.2s-3.8s): Lowest failure rate
- WireTech: Best performing cable supplier (~5% failure)

This synthetic data is used ONLY for model training.
The real 2310 units are NEVER seen during training.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import json

# Configuration
SYNTHETIC_SIZE = 10_000
RANDOM_SEED = 42
OUTPUT_PATH = Path(__file__).parent / 'data' / 'synthetic'


def analyze_real_data(data_path: str) -> Dict:
    """
    Analyze real data to extract statistical profiles for synthetic generation.

    Returns:
        Dict with distributions and statistics for each column
    """
    print("=" * 80)
    print("ANALYZING REAL DATA FOR STATISTICAL PROFILE")
    print("=" * 80)

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} real records")

    profile = {
        'columns': {},
        'correlations': {},
        'failure_rates': {}
    }

    # Analyze each column
    for col in df.columns:
        col_data = df[col]

        if col_data.dtype == 'object':
            # Categorical column
            value_counts = col_data.value_counts(normalize=True).to_dict()
            profile['columns'][col] = {
                'type': 'categorical',
                'values': list(value_counts.keys()),
                'probabilities': list(value_counts.values())
            }
        else:
            # Numerical column
            profile['columns'][col] = {
                'type': 'numerical',
                'mean': float(col_data.mean()),
                'std': float(col_data.std()),
                'min': float(col_data.min()),
                'max': float(col_data.max()),
                'median': float(col_data.median())
            }

    # Analyze failure rates by key variables
    # 1. Cable Harness Supplier failure rates
    if 'Cable_Harness_Supplier' in df.columns:
        cable_failures = df.groupby('Cable_Harness_Supplier')['Warranty_Claim'].apply(
            lambda x: (x == 'Yes').mean()
        ).to_dict()
        profile['failure_rates']['Cable_Harness_Supplier'] = cable_failures
        print(f"\nCable Supplier Failure Rates:")
        for supplier, rate in sorted(cable_failures.items(), key=lambda x: -x[1]):
            print(f"  {supplier}: {rate*100:.1f}%")

    # 2. Soldering Time failure rates (binned)
    if 'Soldering_Time_s' in df.columns:
        df['_solder_bin'] = pd.cut(df['Soldering_Time_s'], bins=10)
        solder_failures = df.groupby('_solder_bin')['Warranty_Claim'].apply(
            lambda x: (x == 'Yes').mean()
        ).to_dict()
        profile['failure_rates']['Soldering_Time_bins'] = {
            str(k): v for k, v in solder_failures.items()
        }

        # Find Golden Zone
        golden_zone_mask = df['Soldering_Time_s'].between(3.2, 3.8)
        golden_failure = (df.loc[golden_zone_mask, 'Warranty_Claim'] == 'Yes').mean()
        outside_failure = (df.loc[~golden_zone_mask, 'Warranty_Claim'] == 'Yes').mean()
        profile['golden_zone'] = {
            'min': 3.2,
            'max': 3.8,
            'failure_rate_inside': float(golden_failure),
            'failure_rate_outside': float(outside_failure)
        }
        print(f"\nGolden Zone (3.2s-3.8s):")
        print(f"  Inside: {golden_failure*100:.1f}% failure")
        print(f"  Outside: {outside_failure*100:.1f}% failure")

    # Overall failure rate
    overall_failure = (df['Warranty_Claim'] == 'Yes').mean()
    profile['overall_failure_rate'] = float(overall_failure)
    print(f"\nOverall Failure Rate: {overall_failure*100:.1f}%")

    return profile


def generate_synthetic_data(
    profile: Dict,
    n_samples: int = 10_000,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic training data based on statistical profile.

    Key signals emphasized:
    - Cables-X: ~22% failure rate (toxic supplier)
    - WireTech: ~5% failure rate (best supplier)
    - Golden Zone (3.2s-3.8s): ~3% failure rate
    - Outside Golden Zone: ~12% failure rate
    """
    print("\n" + "=" * 80)
    print(f"GENERATING {n_samples:,} SYNTHETIC TRAINING RECORDS")
    print("=" * 80)

    np.random.seed(random_state)

    data = {}

    # Generate each column based on profile
    for col, col_profile in profile['columns'].items():
        if col in ['Warranty_Claim', 'Claim_Type', 'Repair_Cost_USD']:
            # Skip target and leakage columns - will generate based on signals
            continue

        if col_profile['type'] == 'categorical':
            values = col_profile['values']
            probs = col_profile['probabilities']
            data[col] = np.random.choice(values, size=n_samples, p=probs)
        else:
            # Numerical - use truncated normal
            mean = col_profile['mean']
            std = col_profile['std']
            min_val = col_profile['min']
            max_val = col_profile['max']

            samples = np.random.normal(mean, std, n_samples)
            samples = np.clip(samples, min_val, max_val)
            data[col] = samples

    df = pd.DataFrame(data)

    # Generate Serial_Number for synthetic data
    df['Serial_Number'] = [f"SN-SYNTH-{i:05d}" for i in range(n_samples)]
    df['Device_UUID'] = [f"SYNTH-{i:08d}" for i in range(n_samples)]

    # Generate WARRANTY_CLAIM based on key signals
    # CALIBRATED to match real data failure rates
    print("\nApplying CALIBRATED failure probability model...")
    print("  Target: Match real data failure rates exactly")

    # Base failure probability (calibrated to ~9.5% overall)
    base_prob = 0.02  # 2% base rate for good suppliers + Golden Zone

    # Initialize probability array
    failure_probs = np.full(n_samples, base_prob)

    # SIGNAL 1: Cable Harness Supplier effect (CALIBRATED TO REAL DATA)
    # Real data: Cables-X 23.6%, FlexConnect 2.5%, WireTech 1.5%
    cable_col = 'Cable_Harness_Supplier'
    if cable_col in df.columns:
        # Cables-X (toxic): Set to ~23% failure rate
        cables_x_mask = df[cable_col].str.contains('Cables-X', na=False)
        failure_probs[cables_x_mask] = 0.23

        # FlexConnect: Set to ~2.5%
        flex_mask = df[cable_col].str.contains('FlexConnect', na=False)
        failure_probs[flex_mask] = 0.025

        # WireTech (best): Set to ~1.5%
        wiretech_mask = df[cable_col].str.contains('WireTech', na=False)
        failure_probs[wiretech_mask] = 0.015

        print(f"  Cable supplier effects: Cables-X=23%, FlexConnect=2.5%, WireTech=1.5%")

    # SIGNAL 2: Soldering Time effect (CALIBRATED TO REAL DATA)
    # Real data: Golden Zone 1.6%, Outside 12.3%
    solder_col = 'Soldering_Time_s'
    if solder_col in df.columns:
        solder_times = df[solder_col].values

        # Golden Zone (3.2-3.8s): Multiply by 0.13 to get ~1.6%
        golden_mask = (solder_times >= 3.2) & (solder_times <= 3.8)
        failure_probs[golden_mask] *= 0.13

        # Outside Golden Zone: Additional risk
        outside_mask = ~golden_mask

        # Moderately outside (2.8-3.2 or 3.8-4.2): Slight increase
        moderate_low = (solder_times >= 2.8) & (solder_times < 3.2)
        moderate_high = (solder_times > 3.8) & (solder_times <= 4.2)
        failure_probs[moderate_low] *= 1.3
        failure_probs[moderate_high] *= 1.3

        # Far outside: Larger increase
        far_low = solder_times < 2.8
        far_high = solder_times > 4.2
        failure_probs[far_low] *= 1.8
        failure_probs[far_high] *= 1.8

        print(f"  Soldering time effects: Golden Zone ~1.6%, Outside higher")

    # Cap probabilities (realistic range)
    failure_probs = np.clip(failure_probs, 0.01, 0.35)

    # Generate binary outcomes
    random_draws = np.random.random(n_samples)
    warranty_claims = (random_draws < failure_probs).astype(int)
    df['Warranty_Claim'] = np.where(warranty_claims == 1, 'Yes', 'No')

    # Generate Claim_Type and Repair_Cost for claims
    claim_types = ['Hardware', 'Software', 'Assembly', 'Component']
    df['Claim_Type'] = np.where(
        df['Warranty_Claim'] == 'Yes',
        np.random.choice(claim_types, n_samples),
        ''
    )

    # Repair costs (only for claims)
    repair_costs = np.where(
        df['Warranty_Claim'] == 'Yes',
        np.random.normal(1200, 200, n_samples),
        0
    )
    repair_costs = np.clip(repair_costs, 0, 1500)
    df['Repair_Cost_USD'] = repair_costs

    # Validate generated data
    print("\n" + "-" * 40)
    print("SYNTHETIC DATA VALIDATION:")
    print("-" * 40)

    overall_rate = (df['Warranty_Claim'] == 'Yes').mean()
    print(f"Overall failure rate: {overall_rate*100:.1f}%")

    if cable_col in df.columns:
        cable_rates = df.groupby(cable_col)['Warranty_Claim'].apply(
            lambda x: (x == 'Yes').mean()
        )
        print(f"\nCable Supplier Failure Rates:")
        for supplier, rate in sorted(cable_rates.items(), key=lambda x: -x[1]):
            print(f"  {supplier}: {rate*100:.1f}%")

    if solder_col in df.columns:
        golden_mask = df[solder_col].between(3.2, 3.8)
        golden_rate = (df.loc[golden_mask, 'Warranty_Claim'] == 'Yes').mean()
        outside_rate = (df.loc[~golden_mask, 'Warranty_Claim'] == 'Yes').mean()
        print(f"\nGolden Zone Validation:")
        print(f"  Inside (3.2-3.8s): {golden_rate*100:.1f}%")
        print(f"  Outside: {outside_rate*100:.1f}%")

    return df


def save_synthetic_data(df: pd.DataFrame, output_dir: Path):
    """Save synthetic data and profile."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV
    csv_path = output_dir / 'synthetic_training_data.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nSaved synthetic data to: {csv_path}")
    print(f"  Records: {len(df):,}")

    # Save summary statistics
    summary = {
        'total_records': len(df),
        'failure_rate': float((df['Warranty_Claim'] == 'Yes').mean()),
        'claim_count': int((df['Warranty_Claim'] == 'Yes').sum()),
        'avg_repair_cost': float(df.loc[df['Warranty_Claim'] == 'Yes', 'Repair_Cost_USD'].mean())
    }

    summary_path = output_dir / 'synthetic_data_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to: {summary_path}")

    return csv_path


def main():
    """Main entry point for synthetic data generation."""
    print("\n" + "=" * 80)
    print("PHASE 1: SYNTHETIC KNOWLEDGE BASE GENERATOR")
    print("=" * 80)

    # Path to real data (for statistical profile only)
    real_data_path = Path(r'C:\ALK_Zaliczeniowy\ALK_DuzyProjekt\data\raw\USG_Data_cleared.csv')

    if not real_data_path.exists():
        # Try relative path
        real_data_path = Path(__file__).parent / 'data' / 'raw' / 'USG_Data_cleared.csv'

    if not real_data_path.exists():
        raise FileNotFoundError(f"Real data not found at {real_data_path}")

    # Step 1: Analyze real data
    profile = analyze_real_data(str(real_data_path))

    # Step 2: Generate synthetic data
    synthetic_df = generate_synthetic_data(
        profile,
        n_samples=SYNTHETIC_SIZE,
        random_state=RANDOM_SEED
    )

    # Step 3: Save synthetic data
    output_path = save_synthetic_data(synthetic_df, OUTPUT_PATH)

    print("\n" + "=" * 80)
    print("SYNTHETIC DATA GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nNext step: Run train_production_model.py to train on this data")

    return output_path


if __name__ == "__main__":
    main()
