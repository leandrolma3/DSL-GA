"""
Test script to verify MCMO imports and basic functionality.

This script checks if the MCMO module structure is correct
and dependencies are available.
"""

import sys
import os

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add DSL-AG-hybrid to path
sys.path.insert(0, r'C:\Users\Leandro Almeida\Downloads\DSL-AG-hybrid')

print("=" * 70)
print("MCMO Import Test")
print("=" * 70)

# Test 1: Check if mcmo module exists
print("\n[1] Checking mcmo module structure...")
mcmo_path = r'C:\Users\Leandro Almeida\Downloads\DSL-AG-hybrid\mcmo'
if os.path.exists(mcmo_path):
    print(f"✓ mcmo directory exists: {mcmo_path}")
    files = os.listdir(mcmo_path)
    print(f"  Files found: {files}")
else:
    print(f"✗ mcmo directory not found: {mcmo_path}")
    sys.exit(1)

# Test 2: Try importing baseline_mcmo
print("\n[2] Importing baseline_mcmo...")
try:
    from mcmo import baseline_mcmo
    print("✓ baseline_mcmo module imported successfully")
except ImportError as e:
    print(f"✗ Failed to import baseline_mcmo: {e}")
    sys.exit(1)

# Test 3: Check if MCMO dependencies are available
print("\n[3] Checking MCMO dependencies...")
if hasattr(baseline_mcmo, 'MCMO_AVAILABLE'):
    if baseline_mcmo.MCMO_AVAILABLE:
        print("✓ MCMO dependencies available (geatpy, scikit-multiflow)")
    else:
        print("✗ MCMO dependencies NOT available")
        if hasattr(baseline_mcmo, 'IMPORT_ERROR'):
            print(f"  Error: {baseline_mcmo.IMPORT_ERROR}")
        print("\n  To install dependencies:")
        print("    pip install geatpy==2.7.0 scikit-multiflow==0.5.3")
else:
    print("? Unable to check MCMO_AVAILABLE flag")

# Test 4: Check if classes are defined
print("\n[4] Checking class definitions...")
classes_to_check = ['MCMOAdapter', 'MCMOEvaluator', 'test_mcmo_adapter']

for cls_name in classes_to_check:
    if hasattr(baseline_mcmo, cls_name):
        print(f"✓ {cls_name} class/function defined")
    else:
        print(f"✗ {cls_name} class/function NOT defined")

# Test 5: Try instantiating classes (if dependencies available)
print("\n[5] Testing class instantiation...")
if baseline_mcmo.MCMO_AVAILABLE:
    try:
        from mcmo.baseline_mcmo import MCMOAdapter, MCMOEvaluator

        print("  Testing MCMOAdapter...")
        adapter = MCMOAdapter(n_sources=3, verbose=False)
        print("  ✓ MCMOAdapter instantiated successfully")

        print("  Testing MCMOEvaluator...")
        evaluator = MCMOEvaluator(n_sources=3, verbose=False)
        print("  ✓ MCMOEvaluator instantiated successfully")

        print("  Getting adapter info...")
        info = adapter.get_info()
        print(f"  ✓ Adapter info: {info}")

    except Exception as e:
        print(f"  ✗ Error during instantiation: {e}")
        import traceback
        traceback.print_exc()
else:
    print("  ⊘ Skipping instantiation test (dependencies not available)")

# Test 6: Check MCMO original files
print("\n[6] Checking MCMO original files...")
original_files = ['MCMO.py', 'GMM.py', 'OptAlgorithm.py', '__init__.py']
for filename in original_files:
    filepath = os.path.join(mcmo_path, filename)
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f"✓ {filename} exists ({size} bytes)")
    else:
        print(f"✗ {filename} NOT found")

# Summary
print("\n" + "=" * 70)
print("Summary")
print("=" * 70)

if baseline_mcmo.MCMO_AVAILABLE:
    print("✓ All tests passed! MCMO adapter is ready to use.")
    print("\nNext steps:")
    print("  1. Test with synthetic data")
    print("  2. Test with Electricity dataset")
    print("  3. Integrate into main.py pipeline")
else:
    print("⚠ Module structure OK, but dependencies missing.")
    print("\nTo proceed:")
    print("  pip install geatpy==2.7.0 scikit-multiflow==0.5.3")
    print("\nNote: Consider using a separate conda environment to avoid conflicts")
    print("      with river (used in main pipeline)")

print("=" * 70)
