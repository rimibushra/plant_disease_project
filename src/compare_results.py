baseline_test_accuracy = 0.0   # replace with your real result
baseline_test_loss = 0.0       # replace with your real result

transfer_test_accuracy = 0.0   # replace with your real result
transfer_test_loss = 0.0       # replace with your real result

print("Model Comparison")
print("-" * 40)

print("Baseline CNN")
print("Test Accuracy:", baseline_test_accuracy)
print("Test Loss:", baseline_test_loss)

print("\nTransfer Learning (MobileNetV2)")
print("Test Accuracy:", transfer_test_accuracy)
print("Test Loss:", transfer_test_loss)

print("\nBest Model:")
if transfer_test_accuracy > baseline_test_accuracy:
    print("Transfer Learning model performed better.")
elif transfer_test_accuracy < baseline_test_accuracy:
    print("Baseline CNN performed better.")
else:
    print("Both models performed equally.")