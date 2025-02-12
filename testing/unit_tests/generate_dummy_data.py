import numpy as np 
import joblib
def generate_dummy_data(num_samples, time_steps):
  """Generates dummy data for magnitude of light curves, predictions, and probed mask.

  Args:
    num_samples: The number of samples to generate.
    time_steps: The number of time steps in each light curve.

  Returns:
    A tuple of three NumPy arrays containing the dummy data for magnitude of light curves, predictions, and probed mask.
  """

  # Generate dummy data for magnitude of light curves.
  light_curves = np.random.normal(size=(num_samples, time_steps))

  # Generate dummy data for predictions.
  predictions = np.random.normal(size=(num_samples, time_steps))

  # Generate dummy data for probed mask.
  probed_mask = np.random.randint(0, 2, size=(num_samples, time_steps)).astype("float")

  return light_curves, predictions, probed_mask

# Example usage:

# Generate dummy data for 100 samples with 100 time steps each.
light_curves, predictions, probed_mask = generate_dummy_data(100, 100)

data = {
    "true": light_curves,
    "predictions": predictions,
    "mask": probed_mask
}


with open("data/test_data/dummy_test_data_r2.joblib", "wb") as f:
    joblib.dump(data, f)
    
    