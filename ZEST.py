
import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# 1. SIMULATED PATIENT
# ==========================================

TRUE_THRESHOLD = 19

def patient_response(stimulus_db):
    return stimulus_db <= TRUE_THRESHOLD


# ==========================================
# 2. ALGORITHM SETTINGS
# ==========================================

min_db = 0
max_db = 40

prior_mean = 28
prior_std = 8

psychometric_slope = 1.5

target_std = 1.5
max_trials = 30

# Possible threshold values
thresholds = np.arange(min_db, max_db + 1, 0.5)


# ==========================================
# 3. INITIAL PROBABILITY DISTRIBUTION
# ==========================================

# Gaussian prior distribution
probability = np.exp(
    -0.5 * ((thresholds - prior_mean) / prior_std)**2
)

# Normalize so probabilities sum to 1
probability /= np.sum(probability)

initial_probability = probability.copy()

# Store simulation data
stimulus_history = []
response_history = []
estimate_history = []
uncertainty_history = []
probability_history = [probability.copy()]


# ==========================================
# 4. PSYCHOMETRIC FUNCTION
# ==========================================

def detection_probability(stimulus, threshold):

    return 1 / (
        1 + np.exp(
            (stimulus - threshold) / psychometric_slope
        )
    )


# ==========================================
# 5. ZEST ALGORITHM
# ==========================================

print("Starting ZEST Test\n")

for trial in range(1, max_trials + 1):

    # Select stimulus using current mean
    stimulus = np.sum(thresholds * probability)

    # Round stimulus to nearest 0.5 dB
    stimulus = round(stimulus * 2) / 2

    # Obtain patient response
    seen = patient_response(stimulus)

    # Calculate likelihood for every threshold
    likelihood = detection_probability(
        stimulus, thresholds
    )

    if not seen:
        likelihood = 1 - likelihood

    # Bayesian probability update
    probability = probability * likelihood

    # Normalize updated distribution
    probability /= np.sum(probability)

    # Calculate estimated threshold
    estimated_threshold = np.sum(
        thresholds * probability
    )

    # Calculate uncertainty
    estimated_std = np.sqrt(
        np.sum(
            probability
            * (thresholds - estimated_threshold)**2
        )
    )

    # Record results
    stimulus_history.append(stimulus)
    response_history.append(seen)
    estimate_history.append(estimated_threshold)
    uncertainty_history.append(estimated_std)
    probability_history.append(probability.copy())

    response = "Seen" if seen else "Not seen"

    print(
        f"Trial {trial}: "
        f"{stimulus:.1f} dB | "
        f"{response} | "
        f"Estimate: {estimated_threshold:.2f} dB | "
        f"SD: {estimated_std:.2f} dB"
    )

    # Check stopping condition
    if estimated_std <= target_std:
        print("\nTarget uncertainty reached.")
        break

else:
    print("\nMaximum presentations reached.")


# ==========================================
# 6. FINAL RESULTS
# ==========================================

error = abs(estimated_threshold - TRUE_THRESHOLD)

print("\n--- FINAL RESULTS ---")
print(f"True threshold: {TRUE_THRESHOLD} dB")
print(f"Estimated threshold: {estimated_threshold:.2f} dB")
print(f"Absolute error: {error:.2f} dB")
print(f"Final uncertainty: {estimated_std:.2f} dB")
print(f"Total presentations: {len(stimulus_history)}")


# ==========================================
# 7. PLOT STIMULUS PRESENTATIONS
# ==========================================

trials = range(1, len(stimulus_history) + 1)

plt.figure(figsize=(9, 5))

plt.plot(
    trials,
    stimulus_history,
    color="gray",
    label="Presented stimulus"
)

for trial, intensity, seen in zip(
    trials, stimulus_history, response_history
):
    color = "green" if seen else "red"
    plt.scatter(trial, intensity, color=color, s=80)

plt.axhline(
    TRUE_THRESHOLD,
    color="blue",
    linestyle="--",
    label="True threshold"
)

plt.plot(
    trials,
    estimate_history,
    color="orange",
    linestyle=":",
    label="Estimated threshold"
)

plt.xlabel("Stimulus Presentation")
plt.ylabel("Sensitivity (dB)")
plt.title("ZEST Threshold Estimation")
plt.xticks(list(trials))
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()


# ==========================================
# 8. PLOT PROBABILITY DISTRIBUTIONS
# ==========================================

plt.figure(figsize=(9, 5))

plt.plot(
    thresholds,
    initial_probability,
    label="Initial distribution",
    linestyle="--"
)

plt.plot(
    thresholds,
    probability,
    label="Final distribution",
    linewidth=2
)

plt.axvline(
    TRUE_THRESHOLD,
    color="green",
    linestyle="--",
    label="True threshold"
)

plt.axvline(
    estimated_threshold,
    color="orange",
    linestyle=":",
    label="Estimated threshold"
)

plt.xlabel("Possible Threshold (dB)")
plt.ylabel("Probability Mass")
plt.title("ZEST Probability Distribution")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()


# ==========================================
# 9. PLOT UNCERTAINTY
# ==========================================

plt.figure(figsize=(9, 5))

plt.plot(
    trials,
    uncertainty_history,
    marker="o",
    label="Standard deviation"
)

plt.axhline(
    target_std,
    color="red",
    linestyle="--",
    label="Stopping criterion"
)

plt.xlabel("Stimulus Presentation")
plt.ylabel("Standard Deviation (dB)")
plt.title("ZEST Uncertainty Reduction")
plt.xticks(list(trials))
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.show()

