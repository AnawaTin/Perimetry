
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

stimulus = 12
step_size = 4
reversals = 0
previous_direction = None

max_reversals = 2
max_trials = 30

min_db = 0
max_db = 40

# Store simulation data
stimulus_history = []
response_history = []
reversal_history = []

estimated_threshold = None


# ==========================================
# 3. STAIRCASE ALGORITHM
# ==========================================

print("Starting 4-2 dB Staircase Test\n")

for trial in range(1, max_trials + 1):

    # Obtain simulated patient response
    seen = patient_response(stimulus)

    # Record results
    stimulus_history.append(stimulus)
    response_history.append(seen)

    # Determine next direction
    if seen:
        direction = 1
        response = "Seen"
    else:
        direction = -1
        response = "Not seen"

    # Detect a reversal
    if (previous_direction is not None
            and direction != previous_direction):

        reversals += 1
        reversal_history.append(stimulus)

        # Switch from 4 dB to 2 dB steps
        step_size = 2

    print(
        f"Trial {trial}: "
        f"{stimulus} dB | "
        f"{response} | "
        f"Step: {step_size} dB | "
        f"Reversals: {reversals}"
    )

    # Check stopping condition
    if reversals >= max_reversals:

        estimated_threshold = (
            reversal_history[-1]
            + reversal_history[-2]
        ) / 2

        break

    # Calculate next stimulus
    next_stimulus = stimulus + direction * step_size

    # Keep stimulus within device range
    next_stimulus = max(min_db, min(max_db, next_stimulus))

    if next_stimulus == stimulus:
        print("Device intensity limit reached.")
        break

    stimulus = next_stimulus
    previous_direction = direction


# ==========================================
# 4. DISPLAY RESULTS
# ==========================================

print("\n--- FINAL RESULTS ---")
print(f"True threshold: {TRUE_THRESHOLD} dB")
print(f"Total presentations: {len(stimulus_history)}")

if estimated_threshold is not None:

    error = abs(estimated_threshold - TRUE_THRESHOLD)

    print(f"Estimated threshold: {estimated_threshold} dB")
    print(f"Absolute error: {error} dB")

else:
    print("Threshold not resolved.")


# ==========================================
# 5. PLOT RESULTS
# ==========================================

trials = list(range(1, len(stimulus_history) + 1))

plt.plot(trials, stimulus_history, color="gray")

for trial, intensity, seen in zip(
        trials, stimulus_history, response_history):

    color = "green" if seen else "red"
    plt.scatter(trial, intensity, color=color, s=80)

plt.axhline(
    TRUE_THRESHOLD,
    linestyle="--",
    color="blue",
    label="True threshold"
)

if estimated_threshold is not None:
    plt.axhline(
        estimated_threshold,
        linestyle=":",
        color="orange",
        label="Estimated threshold"
    )

plt.xlabel("Stimulus Presentation")
plt.ylabel("Stimulus Intensity (dB)")
plt.title("4-2 dB Staircase Simulation")
plt.xticks(trials)
plt.legend()
plt.grid(alpha=0.3)
plt.show()
