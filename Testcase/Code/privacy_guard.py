# import logging
# import sys

# # We try to import numpy, which is essential for numerical operations and noise.
# try:
#     import numpy as np
# except ImportError:
#     logging.basicConfig() # Ensure logging is configured for the error
#     logging.error("="*60)
#     logging.error("FATAL: 'numpy' library not found.")
#     logging.error("Please install it by running:")
#     logging.error("pip install numpy")
#     logging.error("="*60)
#     sys.exit(1) # Exit if the main dependency is missing

# # --- Configuration ---
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - [PRIVACY_GUARD] - %(message)s'
# )

# # --- [ Task 2.3: Differential Privacy Parameters ] ---

# # 1. Privacy Budget (Epsilon):
# # This is the core trade-off.
# # A SMALLER epsilon = MORE privacy, but MORE "noise" (less accurate search).
# # A LARGER epsilon = LESS privacy, but LESS "noise" (more accurate search).
# # 1.0 is a common, strong default.
# PRIVACY_EPSILON = 1.0

# # 2. Probability of Failure (Delta):
# # The probability that our privacy guarantee fails. This should be
# # extremely small, e.g., less than 1 / (size of dataset).
# PRIVACY_DELTA = 1e-5

# # 3. L2 Sensitivity:
# # This is the *most important assumption*. It defines the maximum
# # "change" we expect in a vector if a single piece of data (e.g., one
# # line of code) changes.
# # By using a pre-trained model like CodeBERT, we *assume* its
# # output vectors are (or can be) normalized to have a max L2-norm of 1.
# L2_SENSITIVITY = 1.0

# # --- [ End of Task 2.3 ] ---


# def add_gaussian_noise(vector: np.ndarray, epsilon: float, delta: float, sensitivity: float) -> np.ndarray:
#     """
#     Applies Gaussian noise to a vector to make it differentially private.
#     This is the "Gaussian Mechanism" for differential privacy.

#     Args:
#         vector: The original vector (e.g., a 768-dim CodeBERT embedding).
#         epsilon: The privacy budget (smaller = more noise).
#         delta: The probability of privacy failure (must be small).
#         sensitivity: The L2-sensitivity of the query (we assume 1.0).

#     Returns:
#         A new, "noisy" vector with the same shape as the input.
#     """
#     if epsilon <= 0:
#         logging.error("Epsilon must be positive. Skipping noise addition.")
#         return vector

#     # Calculate the 'scale' (sigma) of the noise to add.
#     # This is the mathematical formula for the Gaussian Mechanism.
#     scale = (sensitivity * np.sqrt(2 * np.log(1.25 / delta))) / epsilon

#     # Generate Gaussian (normal) noise with 0 mean and the calculated scale
#     noise = np.random.normal(0, scale, vector.shape)
    
#     # Add the noise to the original vector
#     noisy_vector = vector + noise
    
#     logging.debug(f"Applied Gaussian noise with scale (sigma): {scale:.4f}")
#     return noisy_vector

# # --- [ Test Function ] ---
# def test_privacy_guard():
#     """
#     Runs a test to show the effect of adding privacy noise.
#     """
#     logging.info("--- [ Testing Differential Privacy Guard ] ---")
    
#     logging.info(f"Parameters: Epsilon={PRIVACY_EPSILON}, Delta={PRIVACY_DELTA}, Sensitivity={L2_SENSITIVITY}")
    
#     # Create a fake 768-dimension vector (e.g., from CodeBERT)
#     # We use np.ones() so we can clearly see the change.
#     fake_vector = np.ones(768)
    
#     logging.info(f"\nOriginal vector (first 5 dims): \n{fake_vector[:5]}")
    
#     # --- Run the privacy function ---
#     noisy_vector = add_gaussian_noise(
#         fake_vector,
#         PRIVACY_EPSILON,
#         PRIVACY_DELTA,
#         L2_SENSITIVITY
#     )
    
#     logging.info(f"\n'Noisy' vector (first 5 dims): \n{noisy_vector[:5]}")
    
#     # Check that the vectors are actually different
#     if not np.array_equal(fake_vector, noisy_vector):
#         logging.info("\nTEST PASSED: Noise was successfully added.")
#         logging.info("The 'noisy' vector is different from the original.")
#     else:
#         logging.warning("\nTEST FAILED: No noise was added.")
        
#     logging.info("\n--- [ Privacy Guard Test Complete ] ---")


# if __name__ == "__main__":
#     # This allows you to run this file directly to test the parameters
#     test_privacy_guard()




import logging
import sys

# We try to import the key library.
# If it's not found, we give a helpful error message and exit.
try:
    import numpy as np
except ImportError:
    logging.basicConfig() # Ensure logging is configured for the error
    logging.error("="*60)
    logging.error("FATAL: 'numpy' library not found.")
    logging.error("This is required for Differential Privacy calculations.")
    logging.error("Please install it by running:")
    logging.error("pip install numpy")
    logging.error("="*60)
    sys.exit(1) # Exit if the dependency is missing

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [PRIVACY_GUARD] - %(message)s'
)

# --- Task 2.3: Define Differential Privacy Parameters ---

# EPSILON (ε): The "Privacy Budget"
# - This is the core parameter of Differential Privacy.
# - A *smaller* epsilon (e.g., 0.1) means *more* privacy (more noise).
# - A *larger* epsilon (e.g., 1.0 or 5.0) means *less* privacy (less noise).
# - This is the "trade-off" dial. We choose a value that provides
#   a strong mathematical guarantee without making the vectors useless.
#   1.0 is a common and reasonable default.
EPSILON = 1.0

# SENSITIVITY (Δf): L1 sensitivity of the query.
# For vector embeddings that are normalized (which CodeBERT's pooler
# output effectively is), the L1 sensitivity is typically 1.
# This value represents the maximum possible change to the query
# output if one individual's data were removed.
SENSITIVITY = 1.0

# The "scale" (b) of the Laplacian noise is calculated from these.
# Scale = Sensitivity / Epsilon
LAPLACE_SCALE = SENSITIVITY / EPSILON
# --- End Configuration ---


def add_laplacian_noise(vector, sensitivity=SENSITIVITY, epsilon=EPSILON):
    """
    Applies Laplacian noise to a NumPy vector to make it
    differentially private.
    
    This is the core function for Task 2.3.
    """
    # Calculate the scale 'b' for the Laplace distribution
    # b = sensitivity / epsilon
    scale = sensitivity / epsilon
    
    # Generate noise from the Laplace distribution
    # The 'size' of the noise must match the 'size' of the vector
    noise = np.random.laplace(loc=0.0, scale=scale, size=vector.shape)
    
    # Add the noise to the original vector
    noisy_vector = vector + noise
    
    return noisy_vector

# --- Self-Test Function ---
def test_privacy_guard():
    """
    Runs a simple test to demonstrate the noise function.
    """
    logging.info("\n--- [ Running Differential Privacy Test ] ---")
    
    # 1. Create a "dummy" vector (e.g., from CodeBERT)
    #    A real vector has 768 dimensions.
    #    We'll use a small one for this test.
    original_vector = np.ones(10) # A simple vector of [1.0, 1.0, ...]
    
    logging.info(f"Original Vector (sample): {original_vector[:10]}...")
    
    # 2. Apply the noise
    noisy_vector = add_laplacian_noise(original_vector, SENSITIVITY, EPSILON)
    
    logging.info(f"'Noisy' Vector (sample):   {noisy_vector[:10]}...")
    
    # 3. Check
    if np.all(original_vector == noisy_vector):
        logging.error("TEST FAILED: Noise was not applied.")
    else:
        logging.info("TEST PASSED: Noise was successfully applied.")
        logging.info(f"(Parameters: Epsilon={EPSILON}, Scale={LAPLACE_SCALE})")
        

# This block runs if you execute the file directly
# (e.g., "python privacy_guard.py")
if __name__ == "__main__":
    test_privacy_guard()

